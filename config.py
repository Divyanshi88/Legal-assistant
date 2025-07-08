"""
Configuration file for OpenRouter RAG pipeline
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 🔐 OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenRouter uses OPENAI_API_KEY format
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# 🤗 Local Embedding Model (used for vector DB)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Output size: 384
  # ✅ Stronger semantic understanding than MiniLM
# Or fallback:
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

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

# 📂 Vector Store & PDF Configuration
CHROMA_PATH = "chroma_store"
PDF_FILE = "data/Womenrights.pdf"

# 🔎 Retrieval Settings
RETRIEVAL_SETTINGS = {
    "search_type": "similarity_score_threshold",
    "k": 5,
    "score_threshold": 0.05  # ✅ Lowered from 0.2 to 0.1 for better matching in small corpus
}

# ✂️ Text Splitting Settings
TEXT_SPLIT_SETTINGS = {
    "chunk_size": 500,
    "chunk_overlap": 100,
    "separators": ["\n\n", "\n", ".", " ", ""]
}

# ✅ Configuration Validator
def validate_config():
    """Validate that required configuration is present"""
    if not OPENROUTER_API_KEY:
        raise ValueError("❌ OPENAI_API_KEY not found in environment variables. Please set your OpenRouter API key.")
    
    if not os.path.exists(PDF_FILE):
        raise FileNotFoundError(f"❌ PDF file not found: {PDF_FILE}")
    
    print("✅ Configuration validated successfully")
    return True

if __name__ == "__main__":
    validate_config()
