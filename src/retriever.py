from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import VECTORSTORE_SAVE_DIRECTORY, EMBEDDING_MODEL_NAME
from src import logger


class VectorStoreRetriever:
    """
    Loads a FAISS vector store from disk and provides an interface for retrieving documents.
    """

    def __init__(self) -> None:
        """
        Initialize the retriever with a specified embedding model.
        """
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    def load_vector_store(self, save_directory: str = VECTORSTORE_SAVE_DIRECTORY) -> None:
        """
        Load the FAISS vector store from the provided directory.

        Args:
            save_directory (str): Directory containing the saved vector store.
        """
        try:
            logger.info("Loading vector store from %s.", save_directory)
            self.vector_store = FAISS.load_local(
                save_directory,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Vector store loaded successfully.")
        except Exception:
            logger.exception("Failed to load vector store.")
            raise

    def retrieve_documents(self) -> object:
        """
        Retrieve documents via a retrieval interface using the Maximal Marginal Relevance (MMR) method.

        Returns:
            object: Interface for retrieving relevant documents.
        """
        try:
            if self.vector_store is None:
                raise ValueError("Vector store is not loaded. Call load_vector_store() first.")
            retriever_interface = self.vector_store.as_retriever(
                #search_type="mmr",
                search_kwargs={
                    "k": 4,  # Number of documents to retrieve.
                    #"lambda_mult": 0.25  # Adjust retrieval diversity.(lower values yield higher diversity)
                }
            )
            logger.info("Document retrieval interface created successfully.")
            return retriever_interface
        except Exception:
            logger.exception("Failed to retrieve documents.")
            raise