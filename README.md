# Cancer RAG (Retrieval-Augmented Generation) System

The Cancer RAG System is a Retrieval-Augmented Generation solution designed to process cancer-related documents, such as the National Cancer Plan, and to enable interactive, context-aware conversations. By extracting and indexing content from PDF files using FAISS and integrating with advanced language models, this system delivers responses to user queries.

## Features

- **PDF Document Processing**: Loads and preprocesses a PDF document, removing unwanted patterns like page numbers, headers, and footers.
- **Text Chunking**: Splits documents into manageable chunks to improve indexing accuracy and retrieval performance.
- **Vector Store Creation**: Uses Pinecone to create a vector store for fast and accurate document retrieval.
- **Conversational AI**: Integrates with OpenAI's GPT-3.5-turbo model to provide conversational responses based on the retrieved documents.
- **Comprehensive Logging**: Implements detailed logging for monitoring system operations and troubleshooting issues.
- **Environment Variables**: Uses environment variables for API keys to enhance security.
- **Error Handling**: Includes extensive error handling for robustness and ease of debugging.

## Algorithms and Techniques Used

### 1. **Text Preprocessing and Chunking**
- **Technique:** The system preprocesses PDF text to remove noise (headers, footers, page numbers) and splits the text into overlapping chunks.
- **Why:** Chunking ensures that each vector represents a coherent piece of information, improving retrieval accuracy. Overlapping helps preserve context across chunk boundaries.
- **Alternatives Considered:** Sentence-level splitting was considered but can lose context for longer answers. Fixed-size chunking with overlap provides a balance between context and retrieval granularity.

### 2. **Vector Embedding and Storage**
- **Algorithm:** Embeddings are generated for each text chunk using OpenAI's embedding models.
- **Vector Store:** Pinecone is used as the vector database for storing and retrieving embeddings.
- **Why Pinecone:** Pinecone offers managed, scalable, and high-performance vector search with minimal setup. Alternatives like FAISS (local, open-source) were considered, but Pinecone's managed service simplifies deployment and scaling.
- **Alternatives:** FAISS is suitable for local or small-scale projects but requires manual scaling and management. Pinecone was chosen for its ease of use and production-readiness.

### 3. **Document Retrieval**
- **Algorithm:** Approximate Nearest Neighbor (ANN) search is used to retrieve the most relevant chunks for a given query.
- **Why ANN:** ANN provides fast and scalable similarity search, which is essential for real-time conversational systems.
- **Alternatives:** Brute-force search is accurate but not scalable for large datasets. ANN (as implemented in Pinecone) offers a good trade-off between speed and accuracy.

### 4. **Conversational Retrieval-Augmented Generation (RAG)**
- **Technique:** The system uses LangChain's `ConversationalRetrievalChain` to combine retrieval with a conversational LLM.
- **Why:** This approach allows the chatbot to answer questions using both retrieved context and chat history, enabling multi-turn, context-aware conversations.
- **Alternatives:** Simple retrieval-then-generation pipelines do not maintain conversation history, leading to less coherent multi-turn interactions.

### 5. **Language Model**
- **Model:** OpenAI's GPT-3.5-turbo (or other specified LLMs) is used for answer generation.
- **Why:** GPT-3.5-turbo provides strong performance for conversational tasks and integrates well with LangChain.
- **Alternatives:** Open-source models (e.g., Llama, GPT-Neo) were considered but may require more resources to deploy and fine-tune for comparable performance.

### 6. **Logging and Monitoring**
- **Technique:** Python's logging module is configured to write logs to a file for traceability and debugging.
- **Why:** File-based logging ensures logs are preserved and not lost in console output, especially when running in environments like Chainlit.

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

5. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory and add your OpenAI and Pinecone API keys and langsmith
   ```text
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT=your_endpoint_here
   LANGSMITH_API_KEY=your_api_key_here
   LANGSMITH_PROJECT=project_name

   OPENAI_API_KEY=your_api_key_here
   PINECONE_API_KEY=your_api_key_here
   ```

6. **Run Data Preprocessing**:
   - Run data_preprocessor.py to create the vector store:
   - Use this when you have new data or at the beginning of the project. If you run multiple times with the same data, it will create duplicates.
   ```bash
   python data_preprocessor.py
   ```
   
7. **Run the Application for CLI**:
   ```bash
   python main.py
   ```

8. **Run the Application for UI**:
   ```bash
   chainlit run chainlit_UI.py
   ```

## Why These Choices?

- **Pinecone over FAISS:** Pinecone is managed, scalable, and production-ready, while FAISS is best for local/small-scale use.
- **Chunking with Overlap:** Preserves context better than sentence-level splitting.
- **Conversational Retrieval Chain:** Maintains chat history for more natural, multi-turn conversations.
- **OpenAI LLMs:** Provide state-of-the-art performance and easy integration.
- **File-based Logging:** Ensures logs are persistent and not lost in UI environments.

## References

- [LangChain Documentation](https://python.langchain.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Chainlit Documentation](https://docs.chainlit.io/)