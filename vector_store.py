"""
Vector store module for the RAG chatbot.

This module handles the creation of embeddings and vector stores for document retrieval.
It supports different embedding models and vector store types.
"""

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import CohereEmbeddings

import config


class VectorStoreManager:
    """A class to manage the creation and interaction with vector stores."""
    
    def __init__(self):
        """Initialize the vector store manager with settings from the config."""
        self._embeddings = None
        self._vector_store = None
    
    @property
    def embeddings(self):
        """Get the embeddings model based on the configuration."""
        if self._embeddings is None:
            embedding_config = config.get_embedding_config()
            provider = embedding_config["provider"]
            
            if provider == "openai":
                self._embeddings = OpenAIEmbeddings(
                    model=embedding_config["model"],
                    openai_api_key=embedding_config["api_key"]
                )
            elif provider == "sentence_transformers":
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_config["model"]
                )
            elif provider == "cohere":
                self._embeddings = CohereEmbeddings(
                    model=embedding_config["model"],
                    cohere_api_key=embedding_config["api_key"]
                )
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")
        
        return self._embeddings
    
    def create_vector_store(self, documents: List[Document]) -> Any:
        """
        Create a vector store from a list of documents.
        
        Args:
            documents: A list of Document objects
            
        Returns:
            A vector store object
        """
        vector_store_config = config.VECTOR_STORE
        vector_store_type = vector_store_config["type"]
        
        if vector_store_type == "faiss":
            self._vector_store = FAISS.from_documents(
                documents, 
                self.embeddings
            )
        elif vector_store_type == "chroma":
            persist_directory = vector_store_config.get("persist_directory")
            self._vector_store = Chroma.from_documents(
                documents, 
                self.embeddings,
                persist_directory=persist_directory
            )
        else:
            raise ValueError(f"Unsupported vector store type: {vector_store_type}")
        
        return self._vector_store
    
    def get_retriever(self):
        """
        Get a retriever for the vector store.
        
        Returns:
            A retriever object
        
        Raises:
            ValueError: If no vector store has been created
        """
        if self._vector_store is None:
            raise ValueError("No vector store has been created. Call create_vector_store() first.")
        
        vector_store_config = config.VECTOR_STORE
        search_type = vector_store_config.get("search_type", "similarity")
        search_k = vector_store_config.get("search_k", 3)
        
        return self._vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": search_k}
        )
    
    def save_vector_store(self, path: Optional[str] = None):
        """
        Save the vector store to disk.
        
        Args:
            path: The path to save the vector store to. If None, uses the default path.
            
        Raises:
            ValueError: If no vector store has been created
        """
        if self._vector_store is None:
            raise ValueError("No vector store has been created. Call create_vector_store() first.")
        
        if path is None:
            path = config.VECTOR_STORE.get("persist_directory", "data/vector_store")
        
        if isinstance(self._vector_store, FAISS):
            self._vector_store.save_local(path)
            print(f"Vector store saved to {path}")
        else:
            # For other vector stores that might have different save methods
            try:
                self._vector_store.persist(path)
                print(f"Vector store persisted to {path}")
            except AttributeError:
                print(f"This vector store type does not support persistence")
    
    def load_vector_store(self, path: Optional[str] = None):
        """
        Load a vector store from disk.
        
        Args:
            path: The path to load the vector store from. If None, uses the default path.
            
        Returns:
            The loaded vector store
            
        Raises:
            FileNotFoundError: If the vector store cannot be loaded
        """
        if path is None:
            path = config.VECTOR_STORE.get("persist_directory", "data/vector_store")
        
        vector_store_type = config.VECTOR_STORE["type"]
        
        try:
            if vector_store_type == "faiss":
                self._vector_store = FAISS.load_local(
                    path, 
                    self.embeddings
                )
            elif vector_store_type == "chroma":
                self._vector_store = Chroma(
                    persist_directory=path,
                    embedding_function=self.embeddings
                )
            else:
                raise ValueError(f"Unsupported vector store type: {vector_store_type}")
            
            print(f"Vector store loaded from {path}")
            return self._vector_store
        
        except Exception as e:
            raise FileNotFoundError(f"Failed to load vector store from {path}: {e}")


# Example usage
# if __name__ == "__main__":
    # from document_loader import DocumentLoader
    
    # # Load some documents
    # loader = DocumentLoader()
    # docs = loader.load_and_split("data/documents/sample.pdf")
    
    # Create a vector store
    # vector_store_manager = VectorStoreManager()
    # vector_store = vector_store_manager.create_vector_store(docs)
    
    # Get a retriever
    # retriever = vector_store_manager.get_retriever()
    
    # Save the vector store
    # vector_store_manager.save_vector_store()
    
    # Load the vector store
    # vector_store_manager.load_vector_store()