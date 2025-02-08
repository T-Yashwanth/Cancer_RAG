import re
import copy
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import path, embedding_model_name, VectorStore_save_directory
from src import logger  # Import the logger

class PDFDocumentHandler:
    def __init__(self, pdf_path=path):
        """
        Initialize the PDFDocumentHandler with the path to the PDF file.

        Args:
            pdf_path (str): The path to the PDF file.
        """
        self.pdf_path = pdf_path
        self.original_documents = None
        self.documents = None

    def load_documents(self):
        """
        Load the documents from the PDF file using PDFPlumberLoader.
        """
        try:
            logger.info(f"Loading documents from {self.pdf_path}.")
            loader = PDFPlumberLoader(self.pdf_path)
            self.original_documents = loader.load()
            self.documents = copy.deepcopy(self.original_documents)
            logger.info("Documents loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load documents: {e}", exc_info=True)
            raise

    def get_original_documents(self):
        """
        Get the original documents loaded from the PDF.

        Returns:
            list: A list of Document objects representing the original documents.
        """
        return self.original_documents

    def get_documents(self):
        """
        Get the copied documents.

        Returns:
            list: A list of Document objects representing the copied documents.
        """
        return self.documents


class TextPreprocessor:
    def __init__(self):
        """
        Initialize the TextPreprocessor with patterns for preprocessing.
        """
        # Define patterns for preprocessing
        self.page_number_pattern = r'Page \d+'  # Page 1
        self.page_range_pattern = r'\d+ of \d+'  # Page 1 of 10
        self.header_footer_pattern = r'NATIONAL CANCER PLAN \| \d'  # NATIONAL CANCER PLAN | 1
        self.whitespace_pattern = r'\s+'

    def preprocess_text(self, text):
        """
        Preprocess the input text by removing unwanted patterns and extra whitespace.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        try:
            # Remove page numbers
            text = re.sub(self.page_number_pattern, '', text)
            text = re.sub(self.page_range_pattern, '', text)

            # Remove repetitive headers/footers
            text = re.sub(self.header_footer_pattern, '', text)

            # Remove extra whitespace and newlines
            text = re.sub(self.whitespace_pattern, ' ', text).strip()
            return text
        except Exception as e:
            logger.error(f"Failed to preprocess text: {e}", exc_info=True)
            raise


class DocumentChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n"]):
        """
        Initialize the DocumentChunker with chunking parameters.

        Args:
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The overlap between chunks.
            separators (list): The separators used for splitting the text.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators
        )

    def chunk_documents(self, documents):
        """
        Split the provided documents into chunks.

        Args:
            documents (list): A list of documents to be chunked.

        Returns:
            list: A list of chunked documents.
        """
        try:
            logger.info("Chunking documents.")
            return self.splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Failed to chunk documents: {e}", exc_info=True)
            raise


class VectorStoreCreator:
    def __init__(self, model_name=embedding_model_name):
        """
        Initialize the VectorStoreCreator with the embedding model.

        Args:
            model_name (str): The name of the HuggingFace embedding model.
        """
        self.model_name = model_name
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)

    def create_vector_store(self, chunks):
        """
        Create a FAISS vector store from the provided chunks.

        Args:
            chunks (list): A list of document chunks to be indexed in the vector store.

        Returns:
            FAISS: A FAISS vector store containing the indexed chunks.
        """
        try:
            logger.info("Creating vector store.")
            return FAISS.from_documents(chunks, self.embedding_model)
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}", exc_info=True)
            raise

    def save_vector_store(self, vector_store, save_directory=VectorStore_save_directory):
        """
        Save the vector store locally to a specified directory.

        Args:
            vector_store (FAISS): The FAISS vector store to save.
            save_directory (str): The directory where the vector store will be saved.
        """
        try:
            logger.info(f"Saving vector store to {save_directory}.")
            vector_store.save_local(save_directory)
            logger.info("Vector store saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}", exc_info=True)
            raise


class DataProcessor:
    def __init__(self):
        """
        Initialize the DataProcessor with the necessary components.
        """
        self.pdf_handler = PDFDocumentHandler()
        self.preprocessor = TextPreprocessor()
        self.chunker = DocumentChunker()
        self.vector_store_creator = VectorStoreCreator()

    def process_data(self):
        """
        Process the data by loading, preprocessing, chunking, and creating a vector store.
        """
        try:
            logger.info("Starting data processing.")

            # Step 1: Load the PDF document
            self.pdf_handler.load_documents()
            documents = self.pdf_handler.get_documents()

            # Step 2: Preprocess the documents
            for doc in documents:
                doc.page_content = self.preprocessor.preprocess_text(doc.page_content)

            # Step 3: Chunk the documents
            chunks = self.chunker.chunk_documents(documents)

            # Step 4: Create and save the vector store
            vector_store = self.vector_store_creator.create_vector_store(chunks)
            self.vector_store_creator.save_vector_store(vector_store, VectorStore_save_directory)

            logger.info("Data processing completed. Vector store saved.")
        except Exception as e:
            logger.error(f"An error occurred during data processing: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    try:
        logger.info("Running data_processor.py as a standalone script.")
        processor = DataProcessor()
        processor.process_data()
    except Exception as e:
        logger.error(f"An error occurred in data_processor.py: {e}", exc_info=True)
        raise