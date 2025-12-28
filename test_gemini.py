import os
from dotenv import load_dotenv
from google import genai

PROMPT = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit('Missing GEMINI_API_KEY. Put it in a .env file like: GEMINI_API_KEY="..."')

    print(f"Using key {api_key[:6]}...")

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=PROMPT,
    )

    print(resp.text)

    usage = resp.usage_metadata
    print(f"Prompt Tokens: {usage.prompt_token_count}")
    print(f"Response Tokens: {usage.candidates_token_count}")

if __name__ == "__main__":
    main()
