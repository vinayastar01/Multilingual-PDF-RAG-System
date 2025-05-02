from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from chunker import chunks


# Step 1: Get all chunks

# Step 2: Load embedding model (Tiny & fast)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

# Step 3: Create FAISS vector store
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Step 4: Save vector store locally
vectorstore.save_local("vectorstore/faiss_index")

# Step 5: Inspect first embedding
sample_text = chunks[0].page_content
embedding = embedding_model.embed_query(sample_text)

# Print results
print(f"[✓] Embedded {len(chunks)} chunks and saved FAISS index.")
print("\nSample Text:\n", sample_text)
print("\nEmbedding Vector (first 10 values):\n", embedding[:10])
print("\nTotal Length of Vector:", len(embedding))
