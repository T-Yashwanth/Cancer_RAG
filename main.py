from src.data_loader import PDFDocumentHandler
from src.data_preprocessing import TextPreprocessor, DocumentChunker, VectorStoreCreator
from src.retriver import VectorStoreRetriever
from src.generator import LLMSetup, Chatbot
from src.config import path, VectorStore_save_directory

def main():
    # Step 1: Load the PDF document
    pdf_handler = PDFDocumentHandler(path)
    pdf_handler.load_documents()
    documents = pdf_handler.get_documents()

    # Step 2: Preprocess the documents
    preprocessor = TextPreprocessor()
    for doc in documents:
        doc.page_content = preprocessor.preprocess_text(doc.page_content)

    # Step 3: Chunk the documents
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)

    # Step 4: Create and save the vector store
    vector_store_creator = VectorStoreCreator()
    vector_store = vector_store_creator.create_vector_store(chunks)
    vector_store_creator.save_vector_store(vector_store, VectorStore_save_directory)

    # Step 5: Load the vector store
    retriever = VectorStoreRetriever()
    retriever.load_vector_store(VectorStore_save_directory)

    # Step 6: Set up the LLM and chatbot
    llm_setup = LLMSetup()
    llm = llm_setup.get_llm()

    # Step 7: Start the chatbot
    chatbot = Chatbot(retriever=vector_store.as_retriever(), llm=llm)
    chatbot.chat()

if __name__ == "__main__":
    main()