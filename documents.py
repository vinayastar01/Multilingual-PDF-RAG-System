import os
from typing import List
import fitz  # PyMuPDF

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader


PDF_FOLDER = "data"  # Change this to your actual folder path


def load_pdf(path: str) -> List[Document]:
    """Smart loader: Choose appropriate loader based on PDF structure."""
    try:
        with fitz.open(path) as doc:
            text = "".join(page.get_text() for page in doc)

        # Detect keywords indicating table-heavy content
        if "table" in text.lower() or "rows" in text.lower():
            print(f"[✓] {os.path.basename(path)} → Possibly table-heavy → Using PDFPlumberLoader")
            return PDFPlumberLoader(path).load()

        # Default to structured text PDF loader
        print(f"[✓] {os.path.basename(path)} → Structured digital → Using PyPDFLoader")
        return PyPDFLoader(path).load()

    except Exception as e:
        print(f"[✗] Failed to load {os.path.basename(path)}: {e}")
        return []


def load_all_pdfs(folder_path: str) -> List[Document]:
    """Load all PDF documents from a folder."""
    all_docs = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            all_docs.extend(load_pdf(full_path))

    return all_docs


docs = load_all_pdfs(PDF_FOLDER)  # ← move outside __main__
if __name__ == "__main__":

    print(f"\nTotal documents loaded: {len(docs)}")

