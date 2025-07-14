# query_database.py - Streamlit optimized version

import os
import time
from typing import List, Dict, Any
import streamlit as st

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables with fallback for deployment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If dotenv is not available (e.g., in some deployment environments)
    # Environment variables should be set directly in the deployment platform
    pass

# Import dependencies with error handling
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.documents import Document
except ImportError as e:
    st.error(f"Missing required dependencies: {e}")
    st.info("Please install: pip install langchain langchain-openai langchain-community chromadb")
    st.stop()

# Configuration - Direct definitions to avoid import issues
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHAT_MODELS = {
    "mistral": "mistralai/mistral-7b-instruct",
    "llama": "meta-llama/llama-3-8b-instruct",
    "gpt": "openai/gpt-3.5-turbo",
    "claude": "anthropic/claude-3-haiku",
    "gemini": "google/gemini-pro"
}

DEFAULT_CHAT_MODEL = "mistralai/mistral-7b-instruct"
CHROMA_PATH = "chroma_db"

RETRIEVAL_SETTINGS = {
    "search_type": "similarity_score_threshold",
    "k": 5,
    "score_threshold": 0.5
}

def validate_config():
    """Validate configuration settings"""
    if not OPENROUTER_API_KEY:
        # Check Streamlit secrets
        try:
            api_key = st.secrets["OPENROUTER_API_KEY"]
            return api_key
        except:
            raise ValueError("❌ OPENROUTER_API_KEY is required. Please set it in Streamlit secrets or .env file.")
    return OPENROUTER_API_KEY

class EnhancedRAGPipeline:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = DEFAULT_CHAT_MODEL
            
        # Validate and get API key
        self.api_key = validate_config()
        self.model_name = model_name
        
        # Initialize components
        self.setup_embeddings()
        self.setup_vectorstore()
        self.setup_llm()
        self.setup_retriever()
        self.setup_prompts()
        self.setup_chains()

    def setup_embeddings(self):
        """Initialize embedding model"""
        try:
            self.embedding = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"✅ Loaded embeddings: {EMBEDDING_MODEL}")
        except Exception as e:
            raise Exception(f"Failed to load embeddings: {e}")

    def setup_vectorstore(self):
        """Initialize vector store"""
        # Check multiple possible locations for the vector store
        possible_paths = [
            CHROMA_PATH,
            "chroma_db",
            os.path.join(os.getcwd(), "chroma_db")
        ]
        
        vector_store_path = None
        for path in possible_paths:
            if os.path.exists(path):
                vector_store_path = path
                break
        
        if not vector_store_path:
            raise FileNotFoundError(f"❌ Vector store not found. Checked paths: {possible_paths}")
        
        try:
            self.vectorstore = Chroma(
                persist_directory=vector_store_path,
                embedding_function=self.embedding
            )
            count = self.vectorstore._collection.count()
            print(f"✅ Loaded vector store from {vector_store_path} with {count} documents")
        except Exception as e:
            raise Exception(f"Failed to load vector store: {e}")

    def setup_llm(self):
        """Initialize language model"""
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
        """Initialize retriever"""
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
        """Initialize prompt templates"""
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
        """Initialize processing chains"""
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
        """Query in legal mode"""
        try:
            return self.legal_chain.invoke(question)
        except Exception as e:
            return f"❌ Error processing legal query: {str(e)}"

    def query_summary(self, question: str) -> str:
        """Query in summary mode"""
        try:
            return self.summary_chain.invoke(question)
        except Exception as e:
            return f"❌ Error processing summary query: {str(e)}"

    def get_sources(self, question: str) -> List[Document]:
        """Get relevant sources for a question"""
        try:
            return self.retriever.get_relevant_documents(question)
        except Exception as e:
            print(f"❌ Error retrieving sources: {e}")
            return []

    def query_with_sources(self, question: str, mode: str = "legal") -> Dict[str, Any]:
        """Query with source information"""
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

# Interactive mode for testing
if __name__ == "__main__":
    print("🟢 Enhanced RAG Pipeline for Women's Rights & Domestic Violence Act")
    print("=" * 60)
    
    try:
        rag = EnhancedRAGPipeline()
        print("✅ Pipeline initialized successfully!")
        
        # Test query
        test_query = "What is domestic violence?"
        result = rag.query_with_sources(test_query)
        print(f"\n📘 Test Query: {test_query}")
        print(f"📋 Answer: {result['answer'][:200]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")