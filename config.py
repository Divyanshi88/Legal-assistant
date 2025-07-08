# config.py

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chat Models
CHAT_MODELS = {
    "mistral": "mistralai/mistral-7b-instruct",
    "llama": "meta-llama/llama-3-8b-instruct",
    "gpt": "openai/gpt-3.5-turbo",
    "claude": "anthropic/claude-3-haiku",
    "gemini": "google/gemini-pro"
}

# Default model
DEFAULT_CHAT_MODEL = "mistralai/mistral-7b-instruct"

# Vector Store Configuration
CHROMA_PATH = "./chroma_db"

# Retrieval Settings
RETRIEVAL_SETTINGS = {
    "search_type": "similarity_score_threshold",
    "k": 5,
    "score_threshold": 0.5
}

# Validation function
def validate_config():
    """Validate configuration settings"""
    if not OPENROUTER_API_KEY:
        raise ValueError("❌ OPENROUTER_API_KEY is required. Please set it in your .env file.")
    
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"❌ Vector store not found at {CHROMA_PATH}. Please run create_database.py first.")
    
    return True

# Debug function
def print_config():
    """Print current configuration (for debugging)"""
    print("🔧 Current Configuration:")
    print(f"   OPENROUTER_API_KEY: {'✅ Set' if OPENROUTER_API_KEY else '❌ Not set'}")
    print(f"   EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    print(f"   DEFAULT_CHAT_MODEL: {DEFAULT_CHAT_MODEL}")
    print(f"   CHROMA_PATH: {CHROMA_PATH}")
    print(f"   RETRIEVAL_SETTINGS: {RETRIEVAL_SETTINGS}")

if __name__ == "__main__":
    print_config()
    try:
        validate_config()
        print("✅ Configuration is valid!")
    except Exception as e:
        print(f"❌ Configuration error: {e}")