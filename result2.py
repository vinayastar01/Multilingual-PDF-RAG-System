from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class RAGPipeline:
    def __init__(self):
        # Initialize embedding model with new package
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Load FAISS vector store with safe metadata handling
        self.vectorstore = FAISS.load_local(
            "vectorstore/faiss_index",
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

        # Initialize retriever with enhanced parameters
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 2,
                "fetch_k": 25,
                "lambda_mult": 0.6
            }
        )

        # Load local LLM with proper config
        self.llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        # Improved generation configuration
        self.generation_config = {
            "max_new_tokens": 350,
            "temperature": 0.7,
            "do_sample": True,  # Fixes the warning
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 4,
            "top_k": 50
        }

    def format_prompt(self, context, query):
        return f"""You are a professional document analyst. Use this context:
        {context}

        Answer this question: {query}
        If the answer isn't in the context, say "This information is not available in the documents."
        Provide page references when possible.
        Answer:"""

    def generate_answer(self, prompt):
        inputs = self.llm_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,  # Increased context window
            truncation=True
        )

        outputs = self.llm_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **self.generation_config
        )

        return self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def query(self, question):
        # Retrieve relevant documents with metadata safeguards
        relevant_docs = self.retriever.invoke(question)

        if not relevant_docs:
            return {
                "question": question,
                "answer": "No relevant documents found",
                "sources": []
            }

        # Safely extract metadata
        sources = []
        context_chunks = []

        for doc in relevant_docs:
            metadata = doc.metadata
            context_chunks.append(doc.page_content)

            source_info = {
                "document": metadata.get("source", "Unknown document"),
                "page": metadata.get("page_number", "N/A"),
                "document_id": metadata.get("document_id", "Unknown ID")
            }
            sources.append(source_info)

        context = "\n\n---\n\n".join(context_chunks)

        # Generate answer
        prompt = self.format_prompt(context, question)
        answer = self.generate_answer(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }


# Usage example
if __name__ == "__main__":
    rag = RAGPipeline()

    query = "I wanted to sleep a little longer, he thought. He had had the same dream that night as a week ago, and once again he had awakened before it ended. this line is taken from which book?"
    result = rag.query(query)

    print(f"Question: {result['question']}")
    print(f"\nAnswer: {result['answer']}")

    if result["sources"]:
        print("\nVerified Sources:")
        for source in result["sources"]:
            print(f"- Document: {source['document']}")
            print(f"  ID: {source['document_id']}")
            print(f"  Page: {source['page']}\n")