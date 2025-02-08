import os

base_directory  = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

path =  os.path.join(base_directory, "data", "national-cancer-plan-508.pdf")

embedding_model_name="all-MiniLM-L6-v2"

VectorStore_save_directory = os.path.join(base_directory, "data")

llm_model = "gpt-3.5-turbo"