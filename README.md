# Cancer RAG (Retrieval-Augmented Generation) System

The Cancer RAG System is a Retrieval-Augmented Generation solution designed to process cancer-related documents, such as the National Cancer Plan, and to enable interactive, context-aware conversations. By extracting and indexing content from PDF files using FAISS and integrating with advanced language models, this system delivers responses to user queries.

## Features

- **PDF Document Processing**: Loads and preprocesses a PDF document, removing unwanted patterns like page numbers, headers, and footers.
- **Text Chunking**: Splits documents into manageable chunks to improve indexing accuracy and retrieval performance.
- **Vector Store Creation**:Uses Pinecone to create a vector store for fast and accurate document retrieval.
- **Conversational AI**: Integrates with OpenAI's GPT-3.5-turbo model to provide conversational responses based on the retrieved documents.
- **Comprehensive Logging**: Implements detailed logging for monitoring system operations and troubleshooting issues.
- **Environment Variables**: Uses environment variables for API keys to enhance security.
- **Error Handling**: Includes extensive error handling for robustness and ease of debugging.

## How It Works

1. **Data Processing**:
   - The `DataProcessor` class loads the PDF document, preprocesses the text, and splits it into chunks.
   - These chunks are then indexed into a Pinecone vector store for efficient retrieval.

2. **Retrieval**:
   - The `VectorStoreRetriever` class loads the vector store and retrieves relevant documents based on user queries.

3. **Conversational AI**:
   - The `Chatbot` class uses OpenAI's GPT-3.5-turbo model to generate responses based on the retrieved documents.
   - The `ConversationalRetrievalChain` integrates the retriever and the language model to provide context-aware responses.

## Code Structure

- **`main.py`**: The entry point of the application. Initializes the data processor, retriever, and chatbot, and starts the conversational loop.
- **`src/data_processor.py`**: Handles loading, preprocessing, chunking, and vector store creation for the PDF document.
- **`src/retriever.py`**: Manages the loading and retrieval of documents from the Pinecone vector store.
- **`src/generator.py`**: Sets up the language model and chatbot, enabling conversational interactions.
- **`src/config.py`**: Contains configuration settings such as file paths, model names, and directories.
- **`src/__init__.py`**: Initializes logging for the project.

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/T-Yashwanth/Cancer_RAG
   cd cancer_rag
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv cancer_rag
   ```

3. **Activate Virtual Environment in gitbash**:
   ```bash
   source cancer_rag/Scripts/activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set Up Environment Variableas**:
   - Create a `.env` file in the root directory and add your OpenAI and Pinecone API keys and langsmith
   ```text
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT=your_endpoint_here
   LANGSMITH_API_KEY=your_api_key_here
   LANGSMITH_PROJECT=project_name

   OPENAI_API_KEY =your_api_key_here
   PINECONE_API_KEY = your_api_key_here
   ```

6. **Run Data Preprocessing**:
   - Run data_preprocessor.py to create the vector store:
   - use this when you have a new date or in the beginning of the prject if you run multiple time with same data it will create duplicates
   ```bash
   python data_preprocessor.py
   ```
   
7. **Run the Application for cli**:
   ```bash
   python main.py
   ```

8. **Run the Application for UI**:
   ```bash
   chainlit run chainlit_UI.py
   ```