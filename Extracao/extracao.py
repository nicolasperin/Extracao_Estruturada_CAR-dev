import base64
import json
import re
import time
from pathlib import Path

from ollama import Client

# =========================
# CONFIG
# =========================
OLLAMA_HOST = "http://localhost:11434"
MODEL = "qwen3.5:9b"

client = Client(host=OLLAMA_HOST)

DOCUMENTO_JSON = Path("OCR/Output/DocumentoOCR.json")
PROMPT_TXT = Path("Extracao/Input/Prompt.txt")
IMG_ERD = Path("Extracao/Input/ERD.jpg")
JSON_MODELO = Path("Extracao/Input/Modelo.json")

OUTPUT_FILE = Path("Extracao/Output/resultado_extracao.json")
RAW_FILE = Path("Extracao/Output/ollama_raw_response.txt")

MAX_DOC_CHARS = 45_000
MAX_PROMPT_CHARS = 10_000

MAX_RETRIES = 4  # tentativas para obter JSON


# =========================
# HELPERS
# =========================
def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"JSON não encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def image_to_b64(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def get_message_text(resp) -> str:
    """
    Normaliza a saída do ollama-python, que pode vir como objeto ou dict.
    Preferimos content. Se não existir, tenta response.
    """
    if hasattr(resp, "message") and resp.message is not None:
        content = (getattr(resp.message, "content", "") or "").strip()
        if content:
            return content
    if isinstance(resp, dict):
        msg = resp.get("message") or {}
        content = (msg.get("content") or "").strip()
        if content:
            return content
        content = (resp.get("response") or "").strip()
        if content:
            return content
    return ""


def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _find_first_json_block(s: str) -> str | None:
    """
    Acha o primeiro bloco JSON válido (objeto {} ou array []) usando pilha.
    Ignora chaves/colchetes dentro de strings.
    """
    s = (s or "").strip()
    if not s:
        return None

    start_positions = [i for i, ch in enumerate(s) if ch in "{["]
    for start in start_positions:
        opener = s[start]
        stack = [opener]
        in_str = False
        esc = False

        for i in range(start + 1, len(s)):
            ch = s[i]

            if in_str:
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = False
                continue

            # fora de string
            if ch == '"':
                in_str = True
                continue

            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    break
                top = stack[-1]
                expected = "}" if top == "{" else "]"
                if ch != expected:
                    break
                stack.pop()
                if not stack:
                    candidate = s[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        break

    return None


def extract_json_from_text(text: str) -> dict | list:
    """
    1) tenta json.loads direto
    2) remove code fences e tenta
    3) encontra primeiro bloco JSON bem-formado e tenta
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Resposta vazia do modelo.")

    try:
        return json.loads(raw)
    except Exception:
        pass

    cleaned = _strip_code_fences(raw)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    block = _find_first_json_block(cleaned)
    if block:
        return json.loads(block)

    raise ValueError("Não foi possível extrair JSON válido da resposta.")


def build_json_schema_from_example(example):
    """
    JSON Schema simples a partir do Modelo.json (exemplo).
    Obs: não vamos usar dict schema no `format` (muitos ollama builds ignoram),
    mas deixamos o schema no prompt para orientar.
    """
    def schema_for(v):
        if v is None:
            return {"type": "null"}
        if isinstance(v, bool):
            return {"type": "boolean"}
        if isinstance(v, int) and not isinstance(v, bool):
            return {"type": "integer"}
        if isinstance(v, float):
            return {"type": "number"}
        if isinstance(v, str):
            return {"type": "string"}
        if isinstance(v, list):
            if v:
                return {"type": "array", "items": schema_for(v[0])}
            return {"type": "array", "items": {}}
        if isinstance(v, dict):
            props = {k: schema_for(val) for k, val in v.items()}
            return {
                "type": "object",
                "properties": props,
                "required": list(v.keys()),
                "additionalProperties": False,
            }
        return {}

    root = schema_for(example)
    if root.get("type") != "object":
        root = {
            "type": "object",
            "properties": {"value": root},
            "required": ["value"],
            "additionalProperties": False,
        }
    return root


def sanity_check_document(doc_text: str):
    """
    Se o documento estiver vazio/curto demais, o modelo tende a alucinar.
    """
    if not doc_text or len(doc_text) < 500:
        raise RuntimeError(
            "DOCUMENTO muito curto/vazio após leitura/truncamento. "
            "Verifique DocumentoOCR.json (talvez não seja texto OCR ou está vazio)."
        )


# =========================
# MAIN
# =========================
def main():
    start = time.perf_counter()

    for f in [DOCUMENTO_JSON, PROMPT_TXT, IMG_ERD, JSON_MODELO]:
        if not f.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {f}")

    prompt_txt = read_text(PROMPT_TXT)[:MAX_PROMPT_CHARS]

    modelo_obj = read_json(JSON_MODELO)
    json_schema = build_json_schema_from_example(modelo_obj)

    documento_small = read_text(DOCUMENTO_JSON)[:MAX_DOC_CHARS]
    sanity_check_document(documento_small)

    erd_b64 = image_to_b64(IMG_ERD)

    # 🔥 Sistema anti-alucinação + JSON only
    system_text = (
        "Você é um extrator de dados para documentos OCR.\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "1) Responda SOMENTE com um JSON válido (nada de texto fora).\n"
        "2) NÃO invente. Se não estiver no documento, use null, \"\", 0, ou [] conforme o tipo.\n"
        "3) Siga estritamente as chaves do schema: não crie campos novos.\n"
        "4) Não inclua coordenadas.\n"
    )

    user_text_base = (
        "TAREFA: extraia do DOCUMENTO conforme o PROMPT.\n"
        "RETORNO: APENAS JSON válido.\n\n"
        "PROMPT:\n" + prompt_txt + "\n\n"
        "SCHEMA (JSON Schema de referência):\n" + json.dumps(json_schema, ensure_ascii=False) + "\n\n"
        "DOCUMENTO (OCR):\n" + documento_small
    )

    # 🔧 Opções para reduzir alucinação
    common_options = {
        "num_keep": 0,
        "temperature": 0.0,
        "top_p": 0.1,
        "repeat_penalty": 1.15,
        "num_ctx": 8192,
    }

    # Conversa base
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text_base, "images": [erd_b64]},
    ]

    last_text = ""
    for attempt in range(1, MAX_RETRIES + 1):
        # ✅ IMPORTANTÍSSIMO: desliga thinking e força "json"
        resp = client.chat(
            model=MODEL,
            messages=messages,
            format="json",
            options=common_options,
        )

        text = get_message_text(resp)
        last_text = text or last_text

        # salva o RAW sempre (pra você inspecionar)
        RAW_FILE.write_text(last_text or "", encoding="utf-8")
        print(f"🧾 RAW salvo em: {RAW_FILE.resolve()} (tentativa {attempt}/{MAX_RETRIES})")

        # tenta extrair json
        try:
            result = extract_json_from_text(last_text)
            OUTPUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            end = time.perf_counter()
            print("✅ OK:", OUTPUT_FILE.resolve())
            print(f"⏱️ {end - start:.2f}s")
            return
        except Exception:
            # retry: manda a resposta errada e ordena converter para JSON do schema
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "A sua resposta anterior NÃO era JSON válido.\n"
                        "CONVERTA para um JSON válido que siga EXATAMENTE o schema.\n"
                        "Regras: somente JSON; sem texto extra; sem inventar; campos ausentes = null/[]/\"\".\n\n"
                        "RESPOSTA ANTERIOR (corrigir):\n"
                        + (last_text[:6000] if last_text else "")
                    ),
                }
            )

    # se chegou aqui, falhou geral
    raise RuntimeError(
        "Falhou em obter JSON válido após retries. "
        f"Veja o arquivo RAW: {RAW_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()