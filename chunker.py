from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from documents import docs  # Import preloaded documents

# Smart splitter based on document source
def get_splitter(source_type: str):
    if source_type == "ocr":
        return CharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            separator=" "  # OCR text lacks punctuation
        )
    elif source_type == "table":
        return RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n", ",", " "]  # Tables may be line- or comma-separated
        )
    else:  # digital or default
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]  # Structured text
        )

# Chunk all docs
def chunk_documents(documents):
    all_chunks = []

    for doc in documents:
        source_type = doc.metadata.get("source", "digital")
        splitter = get_splitter(source_type)

        chunks = splitter.split_documents([doc])

        for idx, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": idx,
                "source": source_type,
            })
            all_chunks.append(chunk)

    return all_chunks


chunks = chunk_documents(docs)
# Run chunking
if __name__ == "__main__":
    print(f"[✓] Total documents loaded: {len(docs)}")


    print(f"[✓] Total chunks generated: {len(chunks)}\n")

    if chunks:
        print("Example chunk:\n")
        print(chunks[0].page_content)
    else:
        print("No chunks found.")
