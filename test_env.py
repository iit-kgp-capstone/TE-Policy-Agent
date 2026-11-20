import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Read configuration
llm_url = os.getenv("LLM_BASE_URL")
llm_model = os.getenv("LLM_MODEL")

embedding_url = os.getenv("EMBEDDING_BASE_URL")
embedding_model = os.getenv("EMBEDDING_MODEL")

# Verify configuration
print("LLM URL:", llm_url)
print("LLM Model:", llm_model)

print("Embedding URL:", embedding_url)
print("Embedding Model:", embedding_model)

# Check API key without displaying it
print("LLM API Key configured:", bool(os.getenv("LLM_API_KEY")))
print("Embedding API Key configured:", bool(os.getenv("EMBEDDING_API_KEY")))
