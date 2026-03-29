import os

from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
RAG_API_URL = os.getenv("RAG_API_URL")
API_KEY = os.getenv("API_KEY")
