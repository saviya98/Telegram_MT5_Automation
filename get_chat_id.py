import os
from telethon import TelegramClient
from dotenv import load_dotenv

load_dotenv()

API_ID = int(os.getenv("TG_API_ID", "0"))
API_HASH = os.getenv("TG_API_HASH", "")
SESSION = os.getenv("TG_SESSION_NAME", "chat_id_session")

if not API_ID or not API_HASH:
    raise SystemExit("Set TG_API_ID and TG_API_HASH in .env first.")

client = TelegramClient(SESSION, API_ID, API_HASH)

async def main():
    name = input("Channel/Group name (as shown in Telegram): ").strip()
    async for dialog in client.iter_dialogs():
        if name.lower() in (dialog.name or "").lower():
            print(f"\nFound: {dialog.name}")
            print(f"Chat ID: {dialog.id}")
            return
    print("\nNot found. Make sure you joined the channel and spelled it correctly.")

if __name__ == "__main__":
    client.start()
    with client:
        client.loop.run_until_complete(main())
