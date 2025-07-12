import os
from openai import OpenAI
from dotenv import load_dotenv

# Lade Umgebungsvariablen
load_dotenv()

client = OpenAI()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

def ask_openai(user_input, context_chunks):
    context_text = "\n\n".join(context_chunks)
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context_text}\n\nQuestion:\n{user_input}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()