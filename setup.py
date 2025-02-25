from setuptools import setup, find_packages

setup(
    name="cancer_rag",
    version="0.3.1",
    description="A Retrieval-Augmented Generation (RAG) system for cancer-related document analysis. Have Chainlit UI, and vector Embeddings are stored in the pinecone db, but logs are displaying in the cli insted of file when using ui and langsmith is working fine",
    author="T-Yashwanth",
    packages=find_packages(),
    python_requires=">=3.8",

)