from langchain_pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import (PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, PINECONE_SEARCH_TYPE, 
                        PINECONE_NAMESPACE, EMBEDDING_MODEL_NAME, PINECONE_DISTANCE_METRICS)
from src import logger
import pinecone

class VectorStoreRetriever:
    """
    Loads a Pinecone vector store and provides an interface for retrieving documents.
    """

    def __init__(self) -> None:
        """
        Initialize the retriever with a specified embedding model and Pinecone index name.
        """
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

    def load_vector_store(self) -> None:
        """
        Load the Pinecone vector store.
        """
        try:
            logger.info("Loading vector store from Pinecone.")
            self.vector_store = Pinecone.from_existing_index(PINECONE_INDEX_NAME, self.embeddings)
            logger.info("Vector store loaded successfully.")
        except Exception:
            logger.exception("Failed to load vector store.")
            raise

    def retrieve_documents(self) -> object:
        """
        Retrieve documents via a retrieval interface using the Specified Search type.

        Returns:
            object: Interface for retrieving relevant documents.
        """
        try:
            if self.vector_store is None:
                raise ValueError("Vector store is not loaded. Call load_vector_store() first.")
            retriever_interface = self.vector_store.as_retriever(
                search_type=PINECONE_SEARCH_TYPE,
                search_kwargs={
                    "k": 4,  # Number of documents to retrieve.
                    "distance_metric": PINECONE_DISTANCE_METRICS  # Specify cosine similarity
                    # "namespace": PINECONE_NAMESPACE  # Optional: Use if you want to namespace your vectors
                }
            )
            logger.info("Document retrieval interface created successfully.")
            return retriever_interface
        except Exception:
            logger.exception("Failed to retrieve documents.")
            raise