import os
from dotenv import load_dotenv
load_dotenv()
class AppConfig:
    CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")
    # EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    SEED_LIMIT = int(os.getenv("SEED_LIMIT", "100"))
    # Twilio
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_MESSAGING_SERVICE_SID = os.getenv("TWILIO_MESSAGING_SERVICE_SID")
    TWILIO_FROM_SMS = os.getenv("TWILIO_FROM_SMS")
    TWILIO_FROM_WHATSAPP = os.getenv("TWILIO_FROM_WHATSAPP")
    TWILIO_VALIDATE_SIGNATURE = os.getenv("TWILIO_VALIDATE_SIGNATURE")
    PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX", "listings-index")
    PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
    PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")  # optional override
    LINE_VALIDATE_SIGNATURE = False

