from src.data_processor import DataProcessor
from src.retriever import VectorStoreRetriever
from src.generator import LLMSetup, Chatbot
from src.config import VECTORSTORE_SAVE_DIRECTORY
from src import logger  # Import the logger

def main() -> None:
    """
    Execute the core pipeline:
    1. (Optional) Reprocess data from the PDF.
    2. Load the FAISS vector store.
    3. Set up the language model.
    4. Initiate the chatbot interactive session.
    """
    try:
        logger.info("Starting the CancerRAG pipeline.")

        # Step 1: Optionally (re)process the PDF data to update the vector store.
        # Uncomment the following lines if data reprocessing is needed.
        # logger.info("Processing PDF data to update vector store.")
        # processor = DataProcessor()
        # processor.process_data()
        
        # Step 2: Load the pre-built vector store from disk.
        logger.info("Loading the vector store from disk.")
        vector_retriever = VectorStoreRetriever()
        vector_retriever.load_vector_store(VECTORSTORE_SAVE_DIRECTORY)
        retriever_interface = vector_retriever.retrieve_documents()

        # Step 3: Initialize the language model.
        logger.info("Initializing the language model.")
        llm_setup = LLMSetup()
        llm = llm_setup.get_llm()

        # Step 4: Instantiate the Chatbot.
        logger.info("Initializing Chatbot.")
        chatbot = Chatbot(retriever=retriever_interface, llm=llm)

        # Interactive loop that handles user input/output.
        print("Chatbot is ready! Type 'exit' to quit.")
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                logger.info("Chatbot session terminated by user.")
                break
            answer = chatbot.get_response(user_input)
            print("Chatbot:", answer)

    except Exception as e:
        logger.error(f"An error occurred in the main pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()