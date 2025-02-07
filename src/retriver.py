from langchain_community.vectorstores import FAISS
from src.config import VectorStore_save_directory, embedding_model_name
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStoreRetriever:
    def __init__(self):
        """
        Initialize the VectorStoreRetriever.
        """
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    def load_vector_store(self, save_directory=VectorStore_save_directory):
        """
        Load a FAISS vector store from a specified directory.

        Args:
            save_directory (str): The directory where the vector store is saved.
        """
        self.vector_store = FAISS.load_local(
            save_directory,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

    def retrieve_documents(self, query, k=5):
        """
        Retrieve relevant documents from the vector store based on a query.

        Args:
            query (str): The query to search for.
            k (int): The number of documents to retrieve.

        Returns:
            list: A list of relevant documents.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not loaded. Call `load_vector_store` first.")
        return self.vector_store.similarity_search(query, k=k)