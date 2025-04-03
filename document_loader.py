"""
Document loader module for the RAG chatbot.

This module handles loading documents from various sources (PDF files, websites, etc.)
and preparing them for processing by the vector store.
"""

import os
from typing import List, Union, Optional
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config


class DocumentLoader:
    """A class to load documents from various sources."""
    
    def __init__(self):
        # Initialize the text splitter with settings from config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.TEXT_PROCESSING["chunk_size"],
            chunk_overlap=config.TEXT_PROCESSING["chunk_overlap"]
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load documents from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            A list of Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        return documents
    
    def load_website(self, url: str) -> List[Document]:
        """
        Load documents from a website.
        
        Args:
            url: URL of the website
            
        Returns:
            A list of Document objects
        """
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        return documents
    def load_text_file(self, file_path: str) -> List[Document]:
        """
        Load documents from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            A list of Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        loader = TextLoader(file_path)
        documents = loader.load()
    
        return documents
    
    def load_and_split(self, source: str, source_type: Optional[str] = None) -> List[Document]:
        """
        Load documents from a source and split them into chunks.
        
        Args:
            source: Path to a file or URL
            source_type: Type of the source ("pdf", "web"). If None, it will be inferred.
            
        Returns:
            A list of split Document objects
        """
        # Infer source type if not provided
        if source_type is None:
            if source.startswith(("http://", "https://")):
                source_type = "web"
            elif source.lower().endswith(".pdf"):
                source_type = "pdf"
            elif source.lower().endswith((".txt", ".text")):
                source_type = "text"
            else:
                raise ValueError(f"Could not infer source type for: {source}")

        # And add the text case to the loading section
        if source_type.lower() == "pdf":
            documents = self.load_pdf(source)
        elif source_type.lower() == "web":
            documents = self.load_website(source)
        elif source_type.lower() == "text":
            documents = self.load_text_file(source)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        # Split the documents
        split_docs = self.text_splitter.split_documents(documents)
        
        return split_docs
    
    def load_directory(self, directory_path: str = None) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory. If None, uses the default from config.
            
        Returns:
            A list of split Document objects
        """
        if directory_path is None:
            directory_path = config.DATA_CONFIG["default_documents_dir"]
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_documents = []
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            if os.path.isfile(file_path) and (filename.lower().endswith(".pdf") or filename.lower().endswith((".txt", ".text"))):
                try:
                    if filename.lower().endswith(".pdf"):
                       docs = self.load_and_split(file_path, "pdf")
                    else:
                        docs = self.load_and_split(file_path, "text")
                    all_documents.extend(docs)
                    print(f"Loaded: {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return all_documents


# # Example usage
# if __name__ == "__main__":
#     loader = DocumentLoader()
    
    # Example: Load a single PDF
    # docs = loader.load_and_split("data/documents/sample.pdf")
    # print(f"Loaded {len(docs)} document chunks")
    
    # Example: Load a website
    # docs = loader.load_and_split("https://example.com")
    # print(f"Loaded {len(docs)} document chunks")
    
    # Example: Load all PDFs from the default directory
    # docs = loader.load_directory()
    # print(f"Loaded {len(docs)} total document chunks from directory")