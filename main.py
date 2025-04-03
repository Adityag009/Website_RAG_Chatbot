"""
Main module for the website-specific RAG chatbot.

This is the entry point for the chatbot application, which ties together
all the components for loading website content, embedding it, and providing
conversational responses.
"""

import os
import argparse
from typing import Optional

from document_loader import DocumentLoader
from vector_store import VectorStoreManager
from llm_service import LLMService
from agent import WebsiteChatbot


def setup_chatbot(website_url: Optional[str] = None, 
                  text_file: Optional[str] = None,
                  use_saved_vectors: bool = False,
                  vector_store_path: Optional[str] = None,
                  website_name: str = "our website") -> WebsiteChatbot:
    """
    Set up all components and create the chatbot.
    
    Args:
        website_url: URL of the website to load (optional if using saved vectors or text file)
        text_file: Path to a text file to load (optional if using website or saved vectors)
        use_saved_vectors: Whether to load a previously saved vector store
        vector_store_path: Path to the saved vector store
        website_name: Name of the website for the chatbot persona
        
    Returns:
        A configured WebsiteChatbot
    """
    print(f"Setting up chatbot for {website_name}...")
    
    # Create the vector store manager
    vector_store_manager = VectorStoreManager()
    
    # Either load saved vectors or create new ones
    if use_saved_vectors and vector_store_path:
        print(f"Loading saved vector store from {vector_store_path}...")
        vector_store_manager.load_vector_store(vector_store_path)
    else:
        loader = DocumentLoader()
        
        if text_file:
            print(f"Loading content from text file {text_file}...")
            docs = loader.load_and_split(text_file, "text")
        elif website_url:
            print(f"Loading content from {website_url}...")
            docs = loader.load_and_split(website_url, "web")
        else:
            raise ValueError("Either website_url, text_file, or use_saved_vectors with vector_store_path must be provided")
        
        print("Creating vector embeddings...")
        vector_store_manager.create_vector_store(docs)
        
        # Save the vector store for future use
        if vector_store_path:
            print(f"Saving vector store to {vector_store_path}...")
            vector_store_manager.save_vector_store(vector_store_path)


def main():
    """Main entry point for the chatbot application."""
    parser = argparse.ArgumentParser(description="Website-specific RAG Chatbot")
    parser.add_argument("--url", type=str, help="URL of the website to load")
    parser.add_argument("--use-saved", action="store_true", help="Use saved vector store")
    parser.add_argument("--vector-path", type=str, default="data/vector_store", help="Path for vector store")
    parser.add_argument("--website-name", type=str, default="our website", help="Name of the website")
    parser.add_argument("--text-file", type=str, help="Path to a text file to load")
    
    args = parser.parse_args()
    
    try:
        # Set up the chatbot
        chatbot = setup_chatbot(
            website_url=args.url,
            text_file=args.text_file,
            use_saved_vectors=args.use_saved,
            vector_store_path=args.vector_path,
            website_name=args.website_name
        )
        
        # Run the interactive chatbot
        print("\n" + "="*50)
        print("Chatbot is ready! Type 'exit', 'quit', or 'bye' to end the conversation.")
        print("="*50 + "\n")
        
        chatbot.run_interactive_session()
        
    except Exception as e:
        print(f"Error starting chatbot: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()