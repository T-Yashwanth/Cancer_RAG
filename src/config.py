import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the base directory as the project root.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths for data resources.
PDF_PATH = os.path.join(BASE_DIR, "data", "national-cancer-plan-508.pdf")
VECTORSTORE_SAVE_DIRECTORY = os.path.join(BASE_DIR, "data")

# Model configuration constants. If you change this you have to change the dimension size at PINECONE_DIMENSIONS as per the model 
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# OpenAI
LLM_MODEL =  "gpt-4.1-mini" #"gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
PINECONE_INDEX_NAME = "cancer-rag"
PINECONE_SEARCH_TYPE = "similarity"  # Maximal Marginal Relevance
#PINECONE_NAMESPACE = "default"  # Optional: Use if you want to namespace your vectors
PINECONE_DIMENSIONS = 384  # Dimensions for the embedding model (all-MiniLM-L6-v2)
PINECONE_SEARCH_TYPE = "similarity"  # Use cosine similarity
PINECONE_DISTANCE_METRICS = "cosine"

Top_K = 4