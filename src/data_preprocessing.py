import re
from src.data_loader import PDFDocumentHandler
from src.config import embedding_model_name, VectorStore_save_directory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings


class TextPreprocessor:
    def __init__(self):
        # Define patterns for preprocessing
        self.page_number_pattern = r'Page \d+'  #Page 1
        self.page_range_pattern = r'\d+ of \d+'  #Page 1 of 10
        self.header_footer_pattern = r'NATIONAL CANCER PLAN \| \d'    #NATIONAL CANCER PLAN | 1
        self.whitespace_pattern = r'\s+'

    def preprocess_text(self, text):
        """
        Preprocess the input text by removing unwanted patterns and extra whitespace.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        # Remove page numbers
        text = re.sub(self.page_number_pattern, '', text)
        text = re.sub(self.page_range_pattern, '', text)

        # Remove repetitive headers/footers
        text = re.sub(self.header_footer_pattern, '', text)

        # Remove extra whitespace and newlines
        text = re.sub(self.whitespace_pattern, ' ', text).strip()

        return text

pre_proces = TextPreprocessor()
for doc in PDFDocumentHandler.get_documents():
    doc.page_content = pre_proces.preprocess_text(doc.page_content)


"""
# Example usage:
preprocessor = TextPreprocessor()
text = "Page 1 NATIONAL CANCER PLAN | 1 This is a sample text. 1 of 10"
cleaned_text = preprocessor.preprocess_text(text)
print(cleaned_text)  # Output: "This is a sample text."
"""

#Chunking the preprocessed document
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
        return self.splitter.split_documents(documents)
    


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
        return FAISS.from_documents(chunks, self.embedding_model)

    def save_vector_store(self, vector_store, save_directory=VectorStore_save_directory):
        """
        Save the vector store locally to a specified directory.

        Args:
            vector_store (FAISS): The FAISS vector store to save.
            save_directory (str): The directory where the vector store will be saved.
        """
        vector_store.save_local(save_directory)
