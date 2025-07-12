import os
import discord_bot
from openai_client import ask_openai
from retriever import Retriever


# Core-Komponenten initialisieren
retriever = Retriever()
retriever.load_or_build(force_rebuild=False)

# Dein zentrales Message-Handler
async def handle_message(query: str, send_func):
    # 1. Relevante Chunks holen
    chunks = retriever.search(query, k=10)
    
#    print("Retrieved chunks:")
#    for idx, chunk in enumerate(chunks, start=1):
#        print(f"--- Chunk {idx} ---")
#        print(chunk)
#        print()
    
    # 2. Prompt an OpenAI senden
    answer = ask_openai(query, chunks)
    # 3. Antwort via send_func zurückschicken
    await send_func(answer)

def main():
    bot = discord_bot.create_bot(handle_message)
    bot.start()

if __name__ == "__main__":
    main()