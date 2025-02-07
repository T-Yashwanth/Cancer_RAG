
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

class LLMSetup:
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        """
        Initialize the LLM setup with a specific model and temperature.

        Args:
            model_name (str): The name of the chat model (e.g., "gpt-3.5-turbo" or "gpt-4").
            temperature (float): The temperature parameter for the chat model.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(temperature=self.temperature, model_name=self.model_name)

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
        self.retriever = retriever
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.conversation_chain = self._create_conversation_chain()

    def _create_conversation_chain(self):
        """
        Create a conversational retrieval chain.

        Returns:
            ConversationalRetrievalChain: The configured conversational retrieval chain.
        """
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory
        )

    def chat(self):
        """
        Start an interactive chat loop with the chatbot.
        """
        print("Chatbot is ready! Type 'exit' to quit.")
        while True:
            user_input = input("User: ")
            if user_input.lower() == "exit":
                break
            # Pass the user query to the conversation chain
            result = self.conversation_chain({"question": user_input})
            print("Chatbot:", result["answer"])