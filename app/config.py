import os

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# LLM Configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")

# Embedding Configuration
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


def validate_llm_config():

    required = {
        "LLM_BASE_URL": LLM_BASE_URL,
        "LLM_API_KEY": LLM_API_KEY,
        "LLM_MODEL": LLM_MODEL
    }

    missing = [
        name for name, value in required.items()
        if not value
    ]

    if missing:
        raise ValueError(
            f"Missing configuration: {', '.join(missing)}"
        )
