import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from src.config import LLM_MODEL, OPENAI_API_KEY
from src import logger



class LLMSetup:
    """
    Sets up the language model with the specified configuration.
    """

    def __init__(self, model_name: str = LLM_MODEL, temperature: float = 0) -> None:
        """
        Initialize the language model with a given model name and temperature.

        Args:
            model_name (str): The OpenAI model name.
            temperature (float): Sampling temperature for response generation.
        """
        try:
            logger.info("Initializing LLM with model '%s' and temperature %s.", model_name, temperature)
            self.model_name = model_name
            self.temperature = temperature
            self.llm = ChatOpenAI(
                temperature=self.temperature,
                model_name=self.model_name,
                openai_api_key= OPENAI_API_KEY
            )
            logger.info("LLM initialized successfully.")
        except Exception:
            logger.exception("Failed to initialize LLM.")
            raise

    def get_llm(self) -> ChatOpenAI:
        """
        Retrieve the initialized language model.

        Returns:
            ChatOpenAI: The language model instance.
        """
        return self.llm


class Chatbot:
    """
    Chatbot that integrates a conversational retrieval chain to generate responses
    based on retrieved documents and the user’s chat history.
    """

    def __init__(self, retriever, llm) -> None:
        """
        Initialize the chatbot with a document retriever interface and a language model.

        Args:
            retriever: An object providing document retrieval.
            llm: An instance of the language model.
        """
        try:
            logger.info("Initializing Chatbot with provided retriever and LLM.")
            self.retriever = retriever
            self.llm = llm
            """self.memory = ConversationSummaryBufferMemory(
                llm=self.llm,  # LLM for summarization
                max_token_limit=1000,  # Maximum token limit for the buffer
                memory_key="chat_history",  # Key to store the conversation history
                return_messages=True,  # Return messages as a list
                output_key="answer"  # Key for the model's response 
            )"""
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            self.conversation_chain = self._create_conversation_chain()
            logger.info("Chatbot initialized successfully.")
        except Exception:
            logger.exception("Failed to initialize Chatbot.")
            raise

    def _create_conversation_chain(self) -> ConversationalRetrievalChain:
        """
        Create the conversational retrieval chain that ties together the language model,
        retriever, and memory for an interactive session.

        Returns:
            ConversationalRetrievalChain: Configured conversational chain.
        """
        try:
            logger.info("Creating conversational retrieval chain.")
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=True
            )
            return chain
        except Exception:
            logger.exception("Failed to create conversational retrieval chain.")
            raise

    def get_response(self, user_query: str) -> str:
        """
        Process a user query through the conversational chain and return the generated answer.

        Args:
            user_query (str): The input query from the user.

        Returns:
            str: The generated answer.
        """
        try:
            logger.info("Processing user query: %s", user_query)
            result = self.conversation_chain.invoke({"question": user_query})
            logger.info("Response from chain includes history and retrieved documents: %s", result)
            answer = result.get("answer")
            logger.info("Generated response: %s", answer)
            return answer
        except Exception:
            logger.exception("An error occurred during response generation.")
            raise