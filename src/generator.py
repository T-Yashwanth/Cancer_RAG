from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from src.config import llm_model
import os
from src import logger  # Import the logger

# Load environment variables from .env file
load_dotenv()

class LLMSetup:
    def __init__(self, model_name=llm_model, temperature=0):
        """
        Initialize the LLM setup with a specific model and temperature.

        Args:
            model_name (str): The name of the chat model (e.g., "gpt-3.5-turbo" or "gpt-4").
            temperature (float): The temperature parameter for the chat model.
        """
        try:
            logger.info(f"Initializing LLM with model: {model_name} and temperature: {temperature}.")
            self.model_name = model_name
            self.temperature = temperature
            self.llm = ChatOpenAI(
                temperature=self.temperature,
                model_name=self.model_name,
                openai_api_key=os.getenv("OPENAI_API_KEY")  # Load API key from .env
            )
            logger.info("LLM initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
            raise

    def get_llm(self):
        """
        Get the initialized LLM.

        Returns:
            ChatOpenAI: The configured chat model.
        """
        return self.llm


class Chatbot:
    def __init__(self, retriever, llm):
        """
        Initialize the Chatbot with a retriever and LLM.

        Args:
            retriever: The retriever object (e.g., FAISS vector store retriever).
            llm: The initialized language model.
        """
        try:
            logger.info("Initializing Chatbot.")
            self.retriever = retriever
            self.llm = llm
            self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
            self.conversation_chain = self._create_conversation_chain()
            logger.info("Chatbot initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Chatbot: {e}", exc_info=True)
            raise

    def _create_conversation_chain(self):
        """
        Create a conversational retrieval chain.

        Returns:
            ConversationalRetrievalChain: The configured conversational retrieval chain.
        """
        try:
            logger.info("Creating conversational retrieval chain.")
            return ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
        except Exception as e:
            logger.error(f"Failed to create conversation chain: {e}", exc_info=True)
            raise

    def chat(self):
        """
        Start an interactive chat loop with the chatbot.
        """
        try:
            logger.info("Starting chat loop.")
            print("Chatbot is ready! Type 'exit' to quit.")
            while True:
                user_input = input("User: ")
                if user_input.lower() == "exit":
                    logger.info("Closing chat loop.")
                    break
                logger.info(f"Processing user query: {user_input}.")
                # Pass only the question to the conversation chain
                result = self.conversation_chain.invoke({"question": user_input})

                # Log the top-k retrieved documents' content
#                if "source_documents" in result:
#                    logger.info("Retrieved source documents:")
#                    for i, doc in enumerate(result["source_documents"], start=1):
#                        logger.info(f"Document {i}: {doc.page_content}")

                logger.info(f"Generated Responce: {result}.")
                print("Chatbot:", result["answer"])
        except Exception as e:
            logger.error(f"An error occurred during chat: {e}", exc_info=True)
            raise