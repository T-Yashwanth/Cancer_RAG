from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import *
from src import logger
from pinecone import Pinecone

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
        self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

    def load_vector_store(self) -> None:
        """
        Load the Pinecone vector store from existing index.
        """
        try:
            logger.info("Loading vector store from Pinecone.")
            self.vector_store = PineconeVectorStore.from_existing_index(
                index_name = PINECONE_INDEX_NAME,
                embedding = self.embeddings
                )
            logger.info("Vector store loaded successfully.")
        except Exception:
            logger.exception("Failed to load vector store.")
            raise

    def retrieve_documents(self) -> object:
        """
        Create retrieval interface with proper search configurations.

        Returns:
            object: Configured retriver Interface
        """
        try:
            if self.vector_store is None:
                raise ValueError("Vector store not loaded. Call load_vector_store() first.")
            retriever_interface = self.vector_store.as_retriever(
                search_type=PINECONE_SEARCH_TYPE,
                search_kwargs={
                    "k": Top_K,  # Number of documents to retrieve.
                    #"distance_metric": PINECONE_DISTANCE_METRICS  # Specify cosine similarity
                    # "namespace": PINECONE_NAMESPACE  # Optional: Use if you want to namespace your vectors
                },
            )
            logger.info("Document retrieval interface created successfully. %s", retriever_interface)
            return retriever_interface
        except Exception:
            logger.exception("Failed to retrieve documents.")
            raise