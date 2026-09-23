import os
from dotenv import load_dotenv

load_dotenv()

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "openai/gpt-oss-20b"

# RAG Hyperparameters
TOP_K = 5
MAX_RETRIEVAL_RETRIES = 3
MAX_GENERATION_RETRIES = 2

# Embeddings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = "chroma_db"

# Logging
LOGS_DIR = "logs"
