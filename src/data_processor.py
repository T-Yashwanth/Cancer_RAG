import re
import copy

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import *
from src import logger

from pinecone import Pinecone, ServerlessSpec

class PDFDocumentHandler:
    """
    Handles loading of a PDF file and maintains both the original and a modifiable copy of the documents.
    """

    def __init__(self, pdf_path: str = PDF_PATH) -> None:
        """
        Initialize with the path to the PDF file.

        Args:
            pdf_path (str): Path to the input PDF document.
        """
        self.pdf_path = pdf_path
        self.original_documents: list[Document] | None = None
        self.documents: list[Document] | None = None

    def load_documents(self) -> None:
        """
        Load the PDF document into memory using PDFPlumberLoader and create a deep copy for processing.
        """
        try:
            logger.info("Loading documents from %s.", self.pdf_path)
            loader = PDFPlumberLoader(self.pdf_path)
            self.original_documents = loader.load()
            self.documents = copy.deepcopy(self.original_documents)
            logger.info("Documents loaded successfully.")
        except Exception:
            logger.exception("Failed to load documents.")
            raise

    def get_original_documents(self) -> list:
        """
        Return the original loaded documents.

        Returns:
            list: Original Document objects.
        """
        return self.original_documents

    def get_documents(self) -> list:
        """
        Return the modifiable copy of the documents.

        Returns:
            list: Copied Document objects.
        """
        return self.documents


class TextPreprocessor:
    """
    Cleans text extracted from documents by removing unwanted artifacts such as page numbers, headers, and extra whitespace.
    """

    def __init__(self) -> None:
        """
        Initialize the preprocessor with regular expressions for text cleaning.
        """
        # Define patterns for preprocessing
        self.page_number_pattern = r'Page \d+'  # Page 1
        self.page_range_pattern = r'\d+ of \d+'  # Page 1 of 10
        self.header_footer_pattern = r'NATIONAL CANCER PLAN \| \d'  # NATIONAL CANCER PLAN | 1
        self.whitespace_pattern = r'\s+'

    def preprocess_text(self, text: str) -> str:
        """
        Remove undesired patterns from the input text.

        Args:
            text (str): Raw text from a document page.

        Returns:
            str: Cleaned text.
        """
        try:
            # Remove isolated page numbers and page range strings.
            text = re.sub(self.page_number_pattern, '', text)
            text = re.sub(self.page_range_pattern, '', text)
            # Remove repetitive header/footer strings.
            text = re.sub(self.header_footer_pattern, '', text)
            # Normalize whitespace.
            text = re.sub(self.whitespace_pattern, ' ', text).strip()
            return text
        except Exception:
            logger.exception("Failed to preprocess text.")
            raise


class DocumentChunker:
    """
    Splits documents into smaller text chunks for more efficient embedding and retrieval.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: list | None = None
    ) -> None:
        """
        Initialize with chunking parameters.

        Args:
            chunk_size (int): Maximum number of characters per chunk.
            chunk_overlap (int): Number of overlapping characters between adjacent chunks.
            separators (list, optional): List of string delimiters for chunk splitting.
                                          Defaults to splitting by double or single newlines.
        """
        if separators is None:
            separators = ["\n\n", "\n"]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators
        )

    def chunk_documents(self, documents: list) -> list:
        """
        Split a list of documents into smaller text chunks.

        Args:
            documents (list): List of Document objects to split.

        Returns:
            list: Chunked Document objects.
        """
        try:
            logger.info("Chunking documents...")
            chunks = self.splitter.split_documents(documents)
            logger.info(f"Documents chunked successfully. Got {len(chunks)} chunks")
            return chunks
        except Exception:
            logger.exception("Failed to chunk documents.")
            raise


class VectorStoreCreator:
    """
    Creates and manages a Pinecone vector store from document chunks.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, index_name: str = PINECONE_INDEX_NAME) -> None:
        """
        Initialize with a specific HuggingFace embedding model and Pinecone index name.

        Args:
            model_name (str): The name of the embedding model.
            index_name (str): The name of the Pinecone index.
        """
        self.model_name = model_name
        self.index_name = index_name
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
        self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        self._ensure_index_exists()

    def _ensure_index_exists(self) -> None:
        """
        Check if the Pinecone index exists. If not, create it.
        """
        if self.index_name not in self.pinecone_client.list_indexes().names():
            logger.info(f"Index '{self.index_name}' not found. Creating new index...")
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=PINECONE_DIMENSIONS,
                metric=PINECONE_DISTANCE_METRICS,
                spec=ServerlessSpec(cloud= PINECONE_CLOUD, region= PINECONE_REGION)
            )
            logger.info(f"Index '{self.index_name}' created successfully.")
        else:
            logger.info(f"Index '{self.index_name}' already exists.")

    def create_vector_store(self, chunks: list) -> PineconeVectorStore:
        """
        Create a Pinecone vector store from the provided document chunks.

        Args:
            chunks (list): A list of document chunks.

        Returns:
            PineconeVectorStore: The created Pinecone vector store.
        """
        try:
            vector_store = PineconeVectorStore.from_documents(
                chunks,
                self.embedding_model,
                index_name=self.index_name
            )
            logger.info("Vector store created successfully.")
            return vector_store
        except Exception:
            logger.exception("Failed to create vector store.")
            raise


class DataProcessor:
    """
    Orchestrates the end-to-end data processing pipeline:
    loading, cleaning, chunking, vector store creation, and persistence.
    """

    def __init__(self) -> None:
        """
        Initialize all processing components.
        """
        self.pdf_handler = PDFDocumentHandler()
        self.preprocessor = TextPreprocessor()
        self.chunker = DocumentChunker()
        self.vector_store_creator = VectorStoreCreator()

    def process_data(self) -> None:
        """
        Execute the data processing steps sequentially:
        1. Load the PDF.
        2. Preprocess the text.
        3. Chunk the documents.
        4. Create and save the Pinecone vector store.
        """
        try:
            logger.info("Starting data processing pipeline.")

            # Load PDF document.
            self.pdf_handler.load_documents()
            documents = self.pdf_handler.get_documents()

            # Preprocess text in each document.
            for doc in documents:
                doc.page_content = self.preprocessor.preprocess_text(doc.page_content)

            # Split documents into smaller chunks.
            chunks = self.chunker.chunk_documents(documents)

            # Create and save the vector store.
            vector_store = self.vector_store_creator.create_vector_store(chunks)

            logger.info("Data processing completed successfully. Vector store is ready.")
        except Exception:
            logger.exception("Data processing failed.")
            raise


if __name__ == "__main__":
    try:
        logger.info("Running data_processor module as standalone script.")
        processor = DataProcessor()
        processor.process_data()
    except Exception:
        logger.exception("Error occurred in data_processor module.")
        raise