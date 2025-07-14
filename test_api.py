import os

# Load environment variables with fallback for deployment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If dotenv is not available (e.g., in some deployment environments)
    # Environment variables should be set directly in the deployment platform
    pass

print("🔑 DEBUG - Loaded API Key:", os.getenv("OPENAI_API_KEY"))
