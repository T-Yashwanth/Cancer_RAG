from setuptools import setup, find_packages

setup(
    name="cancer_rag",
    version="0.2.0",
    description="A Retrieval-Augmented Generation (RAG) system for cancer-related document analysis. Dosent have a UI but borks in CLI and able to track the conversation and history",
    author="T-Yashwanth",
    packages=find_packages(),
    python_requires=">=3.8",

)