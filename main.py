from src.data_processor import DataProcessor
from src.retriver import VectorStoreRetriever
from src.generator import LLMSetup, Chatbot
from src.config import VectorStore_save_directory

def main():
    # Step 1: Process the data (load, preprocess, chunk, and create vector store)
    processor = DataProcessor()
    #processor.process_data()

    # Step 2: Load the vector store
    retriever = VectorStoreRetriever()
    retriever.load_vector_store(VectorStore_save_directory)

    # Step 3: Set up the LLM and chatbot
    llm_setup = LLMSetup()
    llm = llm_setup.get_llm()

    # Step 4: Start the chatbot
    chatbot = Chatbot(retriever=retriever.vector_store.as_retriever(), llm=llm)
    chatbot.chat()

if __name__ == "__main__":
    main()