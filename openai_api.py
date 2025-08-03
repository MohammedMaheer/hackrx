import os
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

async def openai_chat(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=128):
    # messages: list of {"role": "system"|"user"|"assistant", "content": str}
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"OpenAI API error: {e}"
