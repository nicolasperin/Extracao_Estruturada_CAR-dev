import json
from paddleocr import PaddleOCR

def coletar_textos(obj):
    """
    Extrai textos de uma estrutura (dict/list/tuple/str) retornada em pagina.json["res"].
    Funciona para variações comuns do OCRResult.
    """
    textos = []

    def walk(x):
        if x is None:
            return
        if isinstance(x, str):
            # evita pegar strings gigantes "debug"; mantém linhas úteis
            s = x.strip()
            if s:
                textos.append(s)
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
            return
        if isinstance(x, (list, tuple)):
            # padrão comum: [bbox, (text, score)] ou [bbox, text, score]
            if len(x) == 2 and isinstance(x[1], (list, tuple)) and len(x[1]) >= 1 and isinstance(x[1][0], str):
                textos.append(x[1][0].strip())
                return
            for v in x:
                walk(v)
            return

    walk(obj)

    # limpa duplicadas mantendo ordem
    vistos = set()
    out = []
    for t in textos:
        t2 = " ".join(t.split())
        if t2 and t2 not in vistos:
            vistos.add(t2)
            out.append(t2)
    return out

def main():
    entrada = "OCR/Input/Documento.pdf"
    saida = "OCR/Output/Documento_OCR.json"

    ocr = PaddleOCR(
        lang="pt",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    resultados = ocr.predict(entrada)  # no seu caso é LISTA

    paginas = []
    texto_completo = []

    for i, pagina in enumerate(resultados):
        dados = pagina.json or {}
        res = dados.get("res", None)

        textos = coletar_textos(res)

        paginas.append({"pagina": i, "texto": textos})
        texto_completo.extend(textos)

    # opcional: juntar tudo em um único campo também
    saida_obj = {
        "arquivo": entrada,
        "total_paginas": len(paginas),
        "paginas": paginas,
        "texto_completo": "\n".join(texto_completo)
    }

    with open(saida, "w", encoding="utf-8") as f:
        json.dump(saida_obj, f, ensure_ascii=False, indent=2)

    print(f"OK: salvo em {saida}")

if __name__ == "__main__":
    main()