Groq-Powered RAG System
This project demonstrates a Retrieval-Augmented Generation (RAG) system that combines the high-speed inference of Groq's language models with a searchable vector database. The system is designed to provide accurate, context-aware answers by first retrieving relevant information from a specific knowledge source and then using a Large Language Model (LLM) to formulate a detailed response based on that retrieved data.

Key Features
Context-Aware Q&A: The system can answer questions based on a specific set of documents, going beyond the LLM's pre-trained knowledge.

High-Speed Inference: Integrates the Groq API, leveraging its powerful LPU™ inference engine for incredibly fast response times.

Scalable Architecture: Uses a vector store for efficient similarity searches, allowing the system to scale to a large number of documents.

Flexible Data Ingestion: Supports loading data from various sources, including web pages and PDF files.

Interactive Interface: Includes a user-friendly web interface built with Streamlit for easy interaction.

Project Structure
The repository contains the following core components:

groq.ipynb: A Jupyter Notebook that provides a step-by-step breakdown of the core RAG pipeline. It demonstrates how to load documents from a web page, chunk them, create embeddings using OpenAI, store them in a Cassandra/Astra DB vector store, and perform a query using a Groq-powered LLM.

llama3.py: A Streamlit application that builds a Q&A system for PDF documents. It uses PyPDFDirectoryLoader to ingest data from a local directory, creates a FAISS vector store, and employs the Groq API with the Llama3-8b-8192 model to answer user questions.

APP.PY: A second Streamlit application that focuses on web content. This app scrapes a specific web page (https://docs.smith.langchain.com/), processes the documents with RecursiveCharacterTextSplitter and OllamaEmbeddings, stores them in a FAISS vector store, and uses the mixtral-8x7b-32768 model from Groq to handle user queries.

Technologies Used
Groq: Provides the LLM for text generation via its high-performance inference engine.

LangChain: The framework that orchestrates the entire RAG pipeline, from data loading and document splitting to creating the retrieval and generation chains.

Vector Stores:

Cassandra/Astra DB: Used in groq.ipynb for a cloud-based, scalable vector storage solution.

FAISS: Used in llama3.py and APP.PY for a fast, in-memory vector store.

Embedding Models:

OpenAIEmbeddings: Used in groq.ipynb and llama3.py to create vector representations of the documents.

OllamaEmbeddings: Used in APP.PY, providing a way to use local or self-hosted embedding models.

Streamlit: Provides the simple, interactive web interface for the llama3.py and APP.PY applications.

# Context-Aware-Q-A-System-with-LangChain-and-Groq
