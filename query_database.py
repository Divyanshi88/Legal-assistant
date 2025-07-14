# query_database.py - Streamlit optimized version using FAISS
import os
import time
from typing import List, Dict, Any
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document  # 👈 THIS is critical


# Prevent tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # If dotenv is not available

# Import dependencies with error handling
try:
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.documents import Document
except ImportError as e:
    import streamlit as st
    st.error(f"Missing required dependencies: {e}")
    st.info("Please install: pip install langchain langchain-openai langchain-community langchain-core langchain-huggingface faiss-cpu")
    st.stop()

# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHAT_MODEL = "mistralai/mistral-7b-instruct"
FAISS_INDEX_PATH = "faiss_index"

CHAT_MODELS = {
    "mistral": "mistralai/mistral-7b-instruct",
    "llama": "meta-llama/llama-3-8b-instruct",
    "gpt": "openai/gpt-3.5-turbo",
    "claude": "anthropic/claude-3-haiku",
    "gemini": "google/gemini-pro"
}

RETRIEVAL_SETTINGS = {
    "search_type": "similarity_score_threshold",
    "k": 5,
    "score_threshold": 0.5
}

def validate_config():
    if not OPENROUTER_API_KEY:
        try:
            api_key = st.secrets["OPENROUTER_API_KEY"]
            return api_key
        except:
            raise ValueError("❌ OPENROUTER_API_KEY is required. Set it in Streamlit secrets or .env.")
    return OPENROUTER_API_KEY

class EnhancedRAGPipeline:
    def __init__(self, model_name: str = None):
        self.api_key = validate_config()
        self.model_name = model_name or DEFAULT_CHAT_MODEL

        self.setup_embeddings()
        self.setup_vectorstore()
        self.setup_llm()
        self.setup_retriever()
        self.setup_prompts()
        self.setup_chains()

    def setup_embeddings(self):
        try:
            self.embedding = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={
                    "device": "cpu"  # Use CPU by default, change to "cuda" if GPU available
                },
                encode_kwargs={"normalize_embeddings": True}
            )
            print(f"✅ Loaded embeddings: {EMBEDDING_MODEL}")
        except Exception as e:
            raise Exception(f"❌ Failed to load embeddings: {e}")

    def setup_vectorstore(self):
        possible_paths = [
            FAISS_INDEX_PATH,
            "faiss_index",
            os.path.join(os.getcwd(), "faiss_index")
        ]

        vector_store_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if not vector_store_path:
            raise FileNotFoundError(f"❌ FAISS vector store not found. Checked: {possible_paths}")

        try:
            self.vectorstore = FAISS.load_local(
                folder_path=vector_store_path,
                embeddings=self.embedding,
                allow_dangerous_deserialization=True
            )
            print(f"✅ Loaded FAISS vector store from {vector_store_path}")
        except Exception as e:
            raise Exception(f"Failed to load FAISS vector store: {e}")

    def setup_llm(self):
        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=0.0,
                openai_api_key=self.api_key,
                openai_api_base=OPENROUTER_BASE_URL,
                max_tokens=1000
            )
            print(f"✅ Initialized LLM: {self.model_name}")
        except Exception as e:
            raise Exception(f"Failed to initialize LLM: {e}")

    def setup_retriever(self):
        try:
            self.retriever = self.vectorstore.as_retriever(
                search_type=RETRIEVAL_SETTINGS["search_type"],
                search_kwargs={
                    "k": RETRIEVAL_SETTINGS["k"],
                    "score_threshold": RETRIEVAL_SETTINGS["score_threshold"]
                }
            )
            print(f"✅ Configured retriever with k={RETRIEVAL_SETTINGS['k']}")
        except Exception as e:
            raise Exception(f"Failed to setup retriever: {e}")

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
- If information is not in the context, respond with:
"📘 I don't have enough information from the provided documents to answer your question."

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
        try:
            return self.legal_chain.invoke(question)
        except Exception as e:
            return f"❌ Error processing legal query: {str(e)}"

    def query_summary(self, question: str) -> str:
        try:
            return self.summary_chain.invoke(question)
        except Exception as e:
            return f"❌ Error processing summary query: {str(e)}"

    def get_sources(self, question: str) -> List[Document]:
        try:
            return self.retriever.get_relevant_documents(question)
        except Exception as e:
            print(f"❌ Error retrieving sources: {e}")
            return []

    def query_with_sources(self, question: str, mode: str = "legal") -> Dict[str, Any]:
        start_time = time.time()
        try:
            sources = self.get_sources(question)
            context = "\n\n".join(doc.page_content for doc in sources)

            if not context.strip():
                return {
                    "answer": "📘 I don't have enough information from the provided documents to answer your question.",
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
        except Exception as e:
            return {
                "answer": f"❌ Error processing query: {str(e)}",
                "sources": [],
                "processing_time": time.time() - start_time,
                "model": self.model_name,
                "mode": mode
            }


# For standalone testing
if __name__ == "__main__":
    print("🟢 Enhanced RAG Pipeline for Women's Rights & Domestic Violence Act")
    print("=" * 60)

    try:
        rag = EnhancedRAGPipeline()
        print("✅ Pipeline initialized successfully!")

        test_query = "What is domestic violence?"
        result = rag.query_with_sources(test_query)
        print(f"\n📘 Test Query: {test_query}")
        print(f"📋 Answer: {result['answer'][:300]}...")

    except Exception as e:
        print(f"❌ Error: {e}")