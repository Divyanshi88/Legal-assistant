"""
Configuration file for OpenRouter RAG pipeline
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 🔐 OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenRouter uses OpenAI-style keys
OPENROUTER_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")  # Default fallback

# 🤗 Local Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim embeddings

# 💬 Supported Chat Models via OpenRouter
CHAT_MODELS = {
    "mistral": "mistralai/mistral-7b-instruct",
    "mixtral": "mistralai/mixtral-8x7b-instruct",
    "gpt-3.5": "openai/gpt-3.5-turbo",
    "gpt-4": "openai/gpt-4-turbo",
    "claude": "anthropic/claude-3-haiku",
    "llama": "meta-llama/llama-3.1-8b-instruct"
}

# 🎯 Default Model
DEFAULT_CHAT_MODEL = CHAT_MODELS["mistral"]

# 📂 Vector Store & Data
CHROMA_PATH = "chroma_store"
PDF_FILE = "data/Womenrights.pdf"

# 🔎 Retrieval Settings
RETRIEVAL_SETTINGS = {
    "search_type": "similarity_score_threshold",
    "k": 5,
    "score_threshold": 0.05  # Lowered for better recall on small corpus
}

# ✂️ Text Split Settings
TEXT_SPLIT_SETTINGS = {
    "chunk_size": 500,
    "chunk_overlap": 100,
    "separators": ["\n\n", "\n", ".", " ", ""]
}

# ✅ Config Validator
def validate_config():
    if not OPENROUTER_API_KEY:
        raise ValueError("❌ OPENAI_API_KEY not found. Set it in .env or Streamlit secrets.")
    if not os.path.exists(PDF_FILE):
        raise FileNotFoundError(f"❌ PDF file missing: {PDF_FILE}")
    print("✅ Configuration validated successfully")
    return True

# Optional manual test
if __name__ == "__main__":
    validate_config()
