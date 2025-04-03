"""
Agent module for the website-specific RAG chatbot.

This module creates a conversational agent that answers questions specifically about
the website content. The chatbot is designed to be interactive, maintain conversation
context, and provide helpful information about the website only.
"""

from typing import List, Dict, Any, Optional
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_core.retrievers import BaseRetriever
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from llm_service import LLMService


class WebsiteChatbot:
    """A class to manage the website-specific conversational chatbot."""
    
    def __init__(self, llm_service: LLMService, retriever: BaseRetriever, website_name: str = "our website"):
        """
        Initialize the website chatbot.
        
        Args:
            llm_service: An initialized LLMService
            retriever: A document retriever
            website_name: The name of the website (used in prompts)
        """
        self.llm_service = llm_service
        self.retriever = retriever
        self.website_name = website_name
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Generate website information for better responses
        self.website_summary = self._get_website_summary()
        self.website_questions = self._get_website_questions()
        
        # Create the retrieval chain with custom prompt
        self._initialize_chat_chain()
    
    def _get_website_summary(self) -> str:
        """Get a summary of the website content."""
        return self.llm_service.generate_document_summary(self.retriever)
    
    def _get_website_questions(self) -> str:
        """Get possible questions about the website content."""
        return self.llm_service.generate_document_questions(self.retriever)
    
    def _initialize_chat_chain(self):
        """Initialize the retrieval chain with a custom prompt."""
        # Create a system prompt focused on website-specific information
        system_template = f"""You are a helpful customer service assistant for {self.website_name}. 
Your goal is to provide accurate, helpful, and friendly information about our website and services.

IMPORTANT GUIDELINES:
1. ONLY answer questions related to information found on {self.website_name}
2. If asked about topics not covered in our website content, politely explain that you can only provide information about {self.website_name}
3. Do not make up information that isn't in the website content
4. Be concise, friendly, and professional in your responses
5. If you're uncertain about an answer, it's okay to say so
6. Start conversations with a friendly greeting

WEBSITE INFORMATION:
{self.website_summary}

You can help with questions like:
{self.website_questions}

Remember that you represent {self.website_name}, so maintain a helpful and professional tone.
"""
        self.system_prompt = system_template
        
        # Create the document retrieval tool
        self.retrieval_chain = self.llm_service.create_qa_chain(self.retriever)
        
    def get_response(self, query: str) -> str:
        """
        Get a response to a user query about the website.
        
        Args:
            query: The user's query
            
        Returns:
            The chatbot's response as a string
        """
        if not query.strip():
            return "Hi there! How can I help you with information about our website today?"
        
        # Combine the user query with the system prompt to keep the bot focused on website content
        try:
            # Add the conversation history to the context
            chat_history = self.memory.load_memory_variables({})
            history_text = ""
            
            if chat_history and "history" in chat_history:
                history_text = f"Conversation history: {chat_history['history']}\n\n"
            
            # Formulate the query with context
            enhanced_query = f"""{self.system_prompt}

{history_text}User query: {query}

Provide a helpful response using only information from the website content. If the question isn't about our website, politely explain you can only answer questions about the website."""
            
            # Get response via the retrieval chain
            response = self.retrieval_chain({"query": enhanced_query})
            
            # Update conversation memory
            self.memory.save_context(
                {"input": query},
                {"output": response["result"]}
            )
            
            return response["result"]
            
        except Exception as e:
            print(f"Error: {e}")
            # Provide a graceful error response while staying in character
            return "I apologize, but I'm having trouble accessing information about our website right now. Please try asking your question again, or you can reach out to our support team for assistance."
    
    def start_conversation(self):
        """
        Start an interactive conversation with the user.
        
        Returns:
            A greeting message to start the conversation
        """
        greeting = f"👋 Hello! I'm the assistant for {self.website_name}. I can answer questions and provide information about our website, products, and services. How can I help you today?"
        return greeting
            
    def run_interactive_session(self):
        """Run an interactive command-line session with the chatbot."""
        print(self.start_conversation())
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nChatbot: Thank you for chatting! If you have more questions about our website later, feel free to return. Have a great day!")
                break
                
            response = self.get_response(user_input)
            print(f"\nChatbot: {response}")


# Example usage
# if __name__ == "__main__":
#     from document_loader import DocumentLoader
#     from vector_store import VectorStoreManager
#     from llm_service import LLMService
#     import os
    
#     # Example of loading a website
#     print("Loading website content...")
#     loader = DocumentLoader()
#     docs = loader.load_and_split("https://example.com", "web")
    
#     print("Creating vector store...")
#     vector_store_manager = VectorStoreManager()
#     vector_store = vector_store_manager.create_vector_store(docs)
#     retriever = vector_store_manager.get_retriever()
    
#     print("Initializing LLM service...")
#     llm_service = LLMService()
    
#     print("Setting up the website chatbot...")
#     chatbot = WebsiteChatbot(
#         llm_service=llm_service,
#         retriever=retriever,
#         website_name="Example Company"
#     )
    
#     # Run the interactive chatbot
#     chatbot.run_interactive_session()