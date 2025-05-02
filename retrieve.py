from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Step 1: Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

# Step 2: Load the FAISS vector store
vectorstore = FAISS.load_local(
    "vectorstore/faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Step 3: Create retriever with MMR
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Enable Maximal Marginal Relevance
    search_kwargs={
        "k": 3,         # Final results to return
        "fetch_k": 10   # Candidates to consider before reranking
    }
)

# Step 4: Test with a query
query = "in any pdf talking about kashmir?"
results = retriever.get_relevant_documents(query)

# Step 5: Print results
print(f"\nTop {len(results)} MMR results for query:\n")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(doc.page_content)
    print()
