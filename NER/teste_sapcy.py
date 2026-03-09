import spacy

# usar GPU se disponível
try:
    spacy.require_gpu()
    print("GPU ativada")
except:
    print("Rodando na CPU")

# carregar modelo de português
nlp = spacy.load("pt_core_news_lg")

# ler arquivo txt
with open("OCR/output/Documento_OCR.txt", "r", encoding="utf-8") as f:
    texto = f.read()

# processar texto
doc = nlp(texto)

print("\nENTIDADES ENCONTRADAS:\n")

for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")