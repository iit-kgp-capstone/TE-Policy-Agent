from app.config import (
    LLM_BASE_URL,
    LLM_MODEL,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    validate_llm_config
)

# Validate LLM configuration
validate_llm_config()

print("LLM configuration is valid!")

# Display non-secret information
print("LLM URL:", LLM_BASE_URL)
print("LLM Model:", LLM_MODEL)

print("Embedding URL:", EMBEDDING_BASE_URL)
print("Embedding Model:", EMBEDDING_MODEL)

print("API keys are not displayed for security.")
