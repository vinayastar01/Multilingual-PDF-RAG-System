from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr
import torch


class RAGChatbot:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.vectorstore = FAISS.load_local(
            "vectorstore/faiss_index",
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def get_response(self, query):
        docs = self.retriever.invoke(query)
        context = "\n".join([d.page_content for d in docs])
        sources = [
            f"📄 Document: {d.metadata.get('document_id', 'Unknown')} | 📃 Page: {d.metadata.get('page_number', 'N/A')}"
            for d in docs
        ]

        inputs = self.tokenizer(
            f"Answer based on context: {context}\n\nQuestion: {query}",
            return_tensors="pt",
            truncation=True
        )
        outputs = self.model.generate(inputs.input_ids)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer, context, sources

    def format_response(self, answer, context, sources):
        sources_text = "\n".join(sources) if sources else "No sources found"
        return f"""**Answer:** {answer}

<details>
<summary>🧠 Relevant Context</summary>
{context}

🔍 Sources:
{sources_text}
</details>
"""

    def chat_interface(self):
        def respond(message, history):
            answer, context, sources = self.get_response(message)
            return self.format_response(answer, context, sources)

        return gr.ChatInterface(
            respond,
            title="Document Chat Assistant",
            description="Ask questions about your documents",
            examples=["What are the main policies?", "Explain the key findings"],
            theme=gr.themes.Soft(),
            css="""
            details {{
                margin-top: 10px;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                padding: 10px;
            }}
            summary {{
                cursor: pointer;
                font-weight: bold;
            }}
            """
        )


if __name__ == "__main__":
    chatbot = RAGChatbot()
    interface = chatbot.chat_interface()
    interface.launch()