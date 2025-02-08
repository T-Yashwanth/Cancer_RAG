from src.data_processor import DataProcessor
from src.retriver import VectorStoreRetriever
from src.generator import LLMSetup, Chatbot
from src.config import VectorStore_save_directory
from src import logger  # Import the logger

def main():
    """
    Main function to initialize and run the chatbot pipeline.
    Steps include data processing, loading the vector store, setting up the LLM, and starting the chatbot.
    """
    try:
        logger.info("Starting the chatbot pipeline.")

        # Step 1: Process the data (load, preprocess, chunk, and create vector store)
        logger.info("Initializing DataProcessor.")
        processor = DataProcessor()
        #processor.process_data()  # Uncomment to reprocess data if needed

        # Step 2: Load the vector store
        logger.info("Loading vector store.")
        retriever = VectorStoreRetriever()
        retriever.load_vector_store(VectorStore_save_directory)

        # Step 3: Set up the LLM and chatbot
        logger.info("Setting up LLM.")
        llm_setup = LLMSetup()
        llm = llm_setup.get_llm()

        # Step 4: Start the chatbot
        logger.info("Starting chatbot.")
        chatbot = Chatbot(retriever=retriever.vector_store.as_retriever(), llm=llm)
        chatbot.chat()

    except Exception as e:
        logger.error(f"An error occurred in the main pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()