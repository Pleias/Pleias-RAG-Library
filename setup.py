from setuptools import setup, find_packages

setup(
    name="pleias_rag_interface",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "regex>=2023.0.0",
        "vllm==0.23.0",
    ],

    author="Your Name",
    author_email="your.email@example.com",
    description="A simple RAG interface library with citations",
)
