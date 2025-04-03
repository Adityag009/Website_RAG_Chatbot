"""
LLM service module for the RAG chatbot.

This module handles the interaction with various LLM providers and creates
chains for question answering with retrieved documents.
"""

from typing import Dict, Any, List, Optional
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

import config


class LLMService:
    """A class to manage interactions with language models."""
    
    def __init__(self):
        """Initialize the LLM service with settings from the config."""
        self._llm = None
        self._qa_chain = None
    
    @property
    def llm(self):
        """Get the LLM based on the configuration."""
        if self._llm is None:
            llm_config = config.get_llm_config()
            provider = llm_config["provider"]
            model = llm_config["model"]
            temperature = llm_config.get("temperature", 0.7)
            max_tokens = llm_config.get("max_tokens", 1000)
            
            if provider == "openai":
                if model.startswith("gpt"):
                    self._llm = ChatOpenAI(
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        openai_api_key=llm_config["api_key"]
                    )
                else:
                    self._llm = OpenAI(
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        openai_api_key=llm_config["api_key"]
                    )
            elif provider == "gemini":
                self._llm = ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    google_api_key=llm_config["api_key"]
                )
            elif provider == "groq":
                self._llm = ChatGroq(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=llm_config["api_key"]
                )
            elif provider == "anthropic":
                self._llm = ChatAnthropic(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=llm_config["api_key"]
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        
        return self._llm
    
    def create_qa_chain(self, retriever: BaseRetriever) -> RetrievalQA:
        """
        Create a question-answering chain with a retriever.
        
        Args:
            retriever: A document retriever
            
        Returns:
            A RetrievalQA chain
        """
        self._qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Simple chain that "stuffs" all documents into the prompt
            retriever=retriever,
            return_source_documents=True  # Include source documents in the response
        )
        
        return self._qa_chain
    
    def get_response(self, query: str) -> Dict[str, Any]:
        """
        Get a response from the QA chain.
        
        Args:
            query: The user's query
            
        Returns:
            A dictionary containing the response and source documents
            
        Raises:
            ValueError: If no QA chain has been created
        """
        if self._qa_chain is None:
            raise ValueError("No QA chain has been created. Call create_qa_chain() first.")
        
        try:
            response = self._qa_chain({"query": query})
            return response
        except Exception as e:
            print(f"Error getting response: {e}")
            return {"result": f"Error: {str(e)}", "source_documents": []}
    
    def direct_llm_query(self, query: str) -> str:
        """
        Query the LLM directly without using a retriever.
        
        Args:
            query: The user's query
            
        Returns:
            The LLM's response as a string
        """
        try:
            return self.llm.predict(query)
        except Exception as e:
            print(f"Error in direct LLM query: {e}")
            return f"Error: {str(e)}"
    
    def generate_document_summary(self, retriever: BaseRetriever) -> str:
        """
        Generate a summary of the documents in the retriever.
        
        Args:
            retriever: A document retriever
            
        Returns:
            A summary of the documents
        """
        temp_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        try:
            summary_response = temp_chain(
                {"query": "Summarize the document in detail. Do not miss any key points."}
            )
            return summary_response["result"]
        except Exception as e:
            print(f"Error generating document summary: {e}")
            return f"Error: {str(e)}"
    
    def generate_document_questions(self, retriever: BaseRetriever) -> str:
        """
        Generate possible questions from the documents in the retriever.
        
        Args:
            retriever: A document retriever
            
        Returns:
            A list of possible questions as a string
        """
        temp_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        try:
            questions_response = temp_chain(
                {"query": "List all the possible questions based on the given context."}
            )
            return questions_response["result"]
        except Exception as e:
            print(f"Error generating document questions: {e}")
            return f"Error: {str(e)}"


# Example usage
# if __name__ == "__main__":
#     from document_loader import DocumentLoader
#     from vector_store import VectorStoreManager
    
    # Load some documents
    # loader = DocumentLoader()
    # docs = loader.load_and_split("data/documents/sample.pdf")
    
    # Create a vector store and retriever
    # vector_store_manager = VectorStoreManager()
    # vector_store = vector_store_manager.create_vector_store(docs)
    # retriever = vector_store_manager.get_retriever()
    
    # Create an LLM service and QA chain
    # llm_service = LLMService()
    # qa_chain = llm_service.create_qa_chain(retriever)
    
    # Get a response
    # response = llm_service.get_response("What is this document about?")
    # print(response["result"])
    
    # Generate a document summary
    # summary = llm_service.generate_document_summary(retriever)
    # print(summary)