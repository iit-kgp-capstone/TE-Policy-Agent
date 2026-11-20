from app.llm_client import generate_response
import traceback

system_prompt = """
You are an enterprise Travel and Expense Policy Assistant.
Answer accurately and do not invent company policies.
"""

user_prompt = """
Explain what a corporate travel policy is
in three simple sentences.
"""

try:
    print("Connecting to institution LLM...")

    answer = generate_response(
        system_prompt,
        user_prompt
    )

    print("\nLLM CONNECTION SUCCESSFUL")
    print("\nModel response:")
    print(answer)

except Exception as error:
    print("\nLLM CONNECTION FAILED")
    print("Error type:", type(error).__name__)
    print("Error message:", str(error))

    print("\nDetailed error information:")
    traceback.print_exc()

    if error.__cause__:
        print("\nUnderlying cause:")
        print(repr(error.__cause__))