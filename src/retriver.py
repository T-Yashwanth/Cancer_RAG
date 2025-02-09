from langchain_community.vectorstores import FAISS
from src.config import VectorStore_save_directory, embedding_model_name
from langchain_huggingface import HuggingFaceEmbeddings
from src import logger  # Import the logger

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
        try:
            logger.info(f"Loading vector store from {save_directory}.")
            self.vector_store = FAISS.load_local(
                save_directory,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}", exc_info=True)
            raise

    def retrieve_documents(self):
        """
        Retrieve relevant documents from the vector store based on a query.

        Args:
            query (str): The query to search for.
            k (int): The number of documents to retrieve.

        Returns:
            list: A list of relevant documents.
        """
        try:
            if self.vector_store is None:
                raise ValueError("Vector store not loaded. Call `load_vector_store` first.")
            #logger.info(f"Retrieving {k} documents for query: {query}.")
            documents = self.vector_store.as_retriever(
                                        search_type="mmr",
                                        search_kwargs={
                                        "k": 2,           # return 6 documents
                                        "lambda_mult": 0.25  # adjust diversity (lower values yield higher diversity)
                                         }
                )
            
            #print(f"Retrieved {documents} documents.")
            return documents
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}", exc_info=True)
            raise