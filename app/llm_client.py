import truststore
truststore.inject_into_ssl()
from openai import OpenAI

from app.config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL,
    validate_llm_config
)

# Step 1: Validate LLM configuration
validate_llm_config()

# Step 2: Create the LLM client
client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    timeout=60.0,
    max_retries=2
)


# Step 3: Define a function to generate AI responses
def generate_response(system_prompt, user_prompt):

    # Send request to the LLM
    response = client.chat.completions.create(

        model=LLM_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0,
        max_tokens=600
    )

    # Return the generated response
    return response.choices[0].message.content
