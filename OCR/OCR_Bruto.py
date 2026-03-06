from paddleocr import PaddleOCR

def coletar_textos(obj):
    textos = []

    def walk(x):
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                textos.append(s)
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
            return
        if isinstance(x, (list, tuple)):
            if (
                len(x) == 2
                and isinstance(x[1], (list, tuple))
                and len(x[1]) >= 1
                and isinstance(x[1][0], str)
            ):
                textos.append(x[1][0].strip())
                return
            for v in x:
                walk(v)
            return

    walk(obj)

    # remove duplicadas mantendo ordem
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
    saida = "OCR/Output/Documento_OCR.txt"

    ocr = PaddleOCR(
        lang="pt",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    resultados = ocr.predict(entrada)

    texto_completo = []

    for pagina in resultados:
        dados = pagina.json or {}
        res = dados.get("res", None)

        textos = coletar_textos(res)
        texto_completo.extend(textos)

    # salva somente texto bruto
    with open(saida, "w", encoding="utf-8") as f:
        f.write("\n".join(texto_completo))

    print(f"OK: texto bruto salvo em {saida}")


if __name__ == "__main__":
    main()