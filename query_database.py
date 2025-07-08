# test.py

import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

import os
from dotenv import load_dotenv

load_dotenv()  # ✅ Needed locally (has no effect in Streamlit Cloud)

# ✅ Load from environment variables or Streamlit secrets
OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENAI_BASE_URL")

# ✅ Add this check
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables. Please set your OpenRouter API key.")

if not OPENROUTER_BASE_URL:
    raise ValueError("❌ OPENAI_BASE_URL not found in environment variables. Please set your OpenRouter base URL.")


class EnhancedRAGPipeline:
    def __init__(self, model_name: str = DEFAULT_CHAT_MODEL):
        validate_config()
        self.model_name = model_name
        self.setup_embeddings()
        self.setup_vectorstore()
        self.setup_llm()
        self.setup_retriever()
        self.setup_prompts()
        self.setup_chains()

    def setup_embeddings(self):
        self.embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def setup_vectorstore(self):
        if not os.path.exists(CHROMA_PATH):
            raise FileNotFoundError(f"❌ Vector store not found at {CHROMA_PATH}. Please run create_database.py first.")
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embedding
        )
        print(f"✅ Loaded vector store with {self.vectorstore._collection.count()} documents")

    def setup_llm(self):
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0.0,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            max_tokens=1000
        )
        print(f"✅ Initialized LLM: {self.model_name}")

    def setup_retriever(self):
        self.retriever = self.vectorstore.as_retriever(
            search_type=RETRIEVAL_SETTINGS["search_type"],
            search_kwargs={
                "k": RETRIEVAL_SETTINGS["k"],
                "score_threshold": RETRIEVAL_SETTINGS["score_threshold"]
            }
        )

    

    def setup_prompts(self):
            self.legal_prompt_template = """You are Nayya, a specialized legal assistant for Women's Rights and Domestic Violence Act. You provide detailed legal information for professionals.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        GUIDELINES:
        - Use ONLY the provided context to answer questions
        - Provide comprehensive legal analysis with technical terms
        - Include specific legal sections, provisions, and procedures when available
        - Format responses professionally with proper legal language
        - If information is not in the context, respond with:
        "📘 I don't have enough information from the provided documents to answer your question specifically."
        - Always prioritize accuracy and legal precision

        DETAILED LEGAL ANSWER:"""

            self.summary_prompt_template = """You are Nayya, a compassionate legal assistant helping the general public understand Women's Rights and Domestic Violence Act in simple terms.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        GUIDELINES:
        - Use ONLY the provided context to answer questions
        - Explain legal concepts in simple, easy-to-understand language
        - Avoid complex legal jargon - use everyday language
        - Be empathetic and supportive in tone
        - Provide practical, actionable guidance when possible
        - If information is not in the context, respond with:
        "📘 I don't have enough information from the provided documents to answer your question, but I encourage you to seek help from legal professionals or support services."
        - Focus on helping people understand their rights and options

        SIMPLE EXPLANATION:"""

            self.legal_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=self.legal_prompt_template
            )

            self.summary_prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=self.summary_prompt_template
            )


    def setup_chains(self):
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.legal_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.legal_prompt
            | self.llm
            | StrOutputParser()
        )

        self.summary_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | self.summary_prompt
            | self.llm
            | StrOutputParser()
        )

    def query_legal(self, question: str) -> str:
        return self.legal_chain.invoke(question)

    def query_summary(self, question: str) -> str:
        return self.summary_chain.invoke(question)

    def get_sources(self, question: str) -> List[Document]:
        return self.retriever.get_relevant_documents(question)

    def query_with_sources(self, question: str, mode: str = "legal") -> Dict[str, Any]:
        start_time = time.time()
        sources = self.get_sources(question)
        
        context = "\n\n".join(doc.page_content for doc in sources)
        print(f"\n🧠 [DEBUG] Model context:\n{context[:500]}...\n")  # Show first 500 characters

        if not context.strip():
            return {
                "answer": "📘 Answer: I don't have enough information from the provided documents to answer your question.",
                "sources": [],
                "processing_time": time.time() - start_time,
                "model": self.model_name,
                "mode": mode
            }

        answer = self.query_legal(question) if mode == "legal" else self.query_summary(question)
        return {
            "answer": answer,
            "sources": sources,
            "processing_time": time.time() - start_time,
            "model": self.model_name,
            "mode": mode
        }

    
# 👇 INTERACTIVE MODE INCLUDED BELOW 👇

def interactive_query():
    print("🟢 Enhanced RAG Pipeline for Women's Rights & Domestic Violence Act")
    print("=" * 60)

    # Choose model
    print("\n📋 Available Models:")
    for key, model in CHAT_MODELS.items():
        print(f"   {key}: {model}")
    model_choice = input(f"\n🤖 Select model (default: mistral): ").strip().lower()
    selected_model = CHAT_MODELS.get(model_choice, DEFAULT_CHAT_MODEL)

    try:
        print(f"\n🔧 Initializing RAG pipeline with model: {selected_model}...")
        rag = EnhancedRAGPipeline(selected_model)
        print("\n✅ Ready! Ask your legal questions (type 'exit' to quit)")
        print("💡 Commands:")
        print("   • 'legal: <question>' - Detailed legal analysis")
        print("   • 'summary: <question>' - Concise summary")
        print("   • 'sources: <question>' - Show source documents")
        print("   • '<question>' - Default legal mode")

        while True:
            query = input("\n❓ You: ").strip()
            if query.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            if not query:
                continue

            try:
                if query.startswith("legal:"):
                    mode = "legal"
                    question = query[len("legal:"):].strip()
                elif query.startswith("summary:"):
                    mode = "summary"
                    question = query[len("summary:"):].strip()
                elif query.startswith("sources:"):
                    sources = rag.get_sources(query[len("sources:"):].strip())
                    print(f"\n📚 Found {len(sources)} relevant sources:")
                    for i, doc in enumerate(sources, 1):
                        print(f"\n--- Source {i} ---\n{doc.page_content[:500]}")
                    continue
                else:
                    mode = "legal"
                    question = query

                result = rag.query_with_sources(question, mode)
                print(f"\n📘 Answer ({mode} mode):\n{result['answer']}")
                print(f"\n⏱️  Processing time: {result['processing_time']:.2f}s")
                print(f"🤖 Model: {result['model']}")
                print(f"📄 Sources: {len(result['sources'])} documents")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {str(e)}")

if __name__ == "__main__":
    interactive_query()
