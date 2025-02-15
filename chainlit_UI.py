from src.data_processor import DataProcessor
from src.retriever import VectorStoreRetriever
from src.generator import LLMSetup, Chatbot
from src.config import VECTORSTORE_SAVE_DIRECTORY
import chainlit as cl
from src import logger

from langsmith import traceable

async def initialize_app():
    """
    Initialize the application components.
    This function is called once when the app starts.
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

        return chatbot
    except Exception:
        logger.exception("An error occurred during initialization:")
        raise

@cl.on_chat_start
async def start():
    """
    Callback function that is executed once when the chat starts.
    This sends a welcome message to the user.
    """
    chatbot = await initialize_app()
    cl.user_session.set("chatbot", chatbot)
    await cl.Message(content="Chatbot is ready! Type your query to begin.").send()

@traceable(run_type="chain")
@cl.on_message
async def main(message: cl.Message):
    """
    Callback function that handles incoming user messages.
    The user's message is processed and an answer is displayed.
    """
    chatbot = cl.user_session.get("chatbot")
    if chatbot is None:
        await cl.Message(content="Error: Chatbot not initialized.").send()
        return

    try:
        logger.info(f"Received message: {message.content}")

        # Stream the response incrementally
        response = await cl.Message(content="").send()
        full_response = ""
        for chunk in chatbot.get_response(message.content):
            full_response += chunk
            await response.stream_token(chunk)

        await response.update()
        logger.info("Generated response: %s", full_response)
    except Exception:
        logger.exception("An error occurred during response generation.")
        await cl.Message(content="An error occurred. Please try again.").send()

if __name__ == "__main__":
    logger.info("Application started")
    cl.run()