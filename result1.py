from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class RAGPipeline:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.vectorstore = FAISS.load_local(
            "vectorstore/faiss_index",
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 2}
        )

        self.llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def query(self, question):
        # Retrieve documents
        relevant_docs = self.retriever.invoke(question)

        # Collect context and sources
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        sources = [
            {
                "document_id": doc.metadata.get("document_id", "Unknown"),
                "page": doc.metadata.get("page_number", "N/A"),
                "source": doc.metadata.get("source", "Unknown")
            }
            for doc in relevant_docs
        ]

        # Create final prompt
        prompt = f"""
        Context from documents:
        {context}

        Question: {question}

        Answer based only on the context. If answer isn't in context, say "I don't know".
        Answer:
        """

        # Generate answer
        inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = self.llm_model.generate(inputs.input_ids)
        answer = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "question": question,
            "context": context,
            "prompt": prompt,
            "answer": answer,
            "sources": sources
        }


# Usage
if __name__ == "__main__":
    rag = RAGPipeline()
    result = rag.query("I wanted to sleep a little longer, he thought. He had had the same dream that night as a week ago, and once again he had awakened before it ended. this line is taken from which book?")

    print("=== CONTEXT ===")
    print(result["context"])

    print("\n=== FINAL PROMPT ===")
    print(result["prompt"])

    print("\n=== ANSWER ===")
    print(result["answer"])

    print("\n=== SOURCES ===")
    for src in result["sources"]:
        print(f"Document: {src['document_id']}, Page: {src['page']}")