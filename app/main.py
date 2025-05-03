from pathlib import Path
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def extrair_texto_pdf(caminho_pdf: Path) -> str:
    doc = fitz.open(str(caminho_pdf))
    texto = ""
    for pagina in doc:
        texto += pagina.get_text()
    return texto


def dividir_em_chunks(texto: str, tamanho_max_palavras: int = 500):
    palavras = texto.split()
    chunks = []
    for i in range(0, len(palavras), tamanho_max_palavras):
        chunk = " ".join(palavras[i:i + tamanho_max_palavras])
        chunks.append(chunk)
    return chunks


def gerar_embeddings(chunks, modelo):
    return modelo.encode(chunks)


def criar_index_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def buscar_chunks(pergunta, modelo, index, chunks, top_k=3):
    embedding_pergunta = modelo.encode([pergunta])
    distancias, indices = index.search(np.array(embedding_pergunta), top_k)
    return [chunks[i] for i in indices[0]]


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    caminho_pdf = BASE_DIR / "input" / "meu-curriculo.pdf"

    print("📄 Lendo PDF...")
    texto = extrair_texto_pdf(caminho_pdf)
    chunks = dividir_em_chunks(texto)

    print("🔎 Gerando embeddings...")
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = gerar_embeddings(chunks, modelo)

    print("📦 Criando índice FAISS...")
    index = criar_index_faiss(np.array(embeddings))

    print("✅ Sistema pronto! Digite sua pergunta:")
    while True:
        pergunta = input("\n❓ Pergunta (ou 'sair'): ")
        if pergunta.lower() in ["sair", "exit", "q"]:
            break
        resultados = buscar_chunks(pergunta, modelo, index, chunks)
        for i, r in enumerate(resultados, 1):
            print(f"\n--- Chunk {i} ---\n{r}")
