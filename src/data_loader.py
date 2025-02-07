from langchain_community.document_loaders import PDFPlumberLoader
from src.config import path
import copy



class PDFDocumentHandler:
    def __init__(self, pdf_path):
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
        loader = PDFPlumberLoader(self.pdf_path)
        self.original_documents = loader.load()
        self.documents = copy.deepcopy(self.original_documents)

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

"""# Example usage:
pdf_handler = PDFDocumentHandler(pdf_path)
pdf_handler.load_documents()

original_docs = pdf_handler.get_original_documents()
copied_docs = pdf_handler.get_documents()

"""
"""

pdf_path = path
loader = PDFPlumberLoader(pdf_path)
Original_documents = loader.load()

documents = copy.deepcopy(Original_documents)"""