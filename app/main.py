from pathlib import Path
import fitz  # PyMuPDF


def extrair_texto_pdf(caminho_pdf: str) -> str:
    doc = fitz.open(caminho_pdf)
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


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    caminho_pdf = BASE_DIR / "input" / "meu-curriculo.pdf"
    texto = extrair_texto_pdf(caminho_pdf)
    chunks = dividir_em_chunks(texto)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---\n{chunk}\n")
