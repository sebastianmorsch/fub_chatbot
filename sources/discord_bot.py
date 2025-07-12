import discord
import os
from dotenv import load_dotenv

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

def create_bot(on_message_callback):
    # Initialize intents and client
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"We have logged in as {client.user}")

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return

        text = message.content.strip()
        if not text:
            return

        # Pass send_func to the callback
        async def send_func(response_text: str):
            await message.channel.send(response_text)

        # Only the text and send function are passed here
        await on_message_callback(text, send_func)

    # We return a simple interface
    class Bot:
        def start(self):
            client.run(DISCORD_TOKEN)

    return Bot()