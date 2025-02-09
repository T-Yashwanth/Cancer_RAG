import os

# Define the base directory as the project root.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Paths for data resources.
PDF_PATH = os.path.join(BASE_DIR, "data", "national-cancer-plan-508.pdf")
VECTORSTORE_SAVE_DIRECTORY = os.path.join(BASE_DIR, "data")

# Model configuration constants.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt-3.5-turbo"