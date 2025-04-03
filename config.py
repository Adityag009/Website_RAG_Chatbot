"""
Configuration settings for the RAG chatbot.

This module contains all configuration parameters that can be adjusted to customize
the behavior of the chatbot, including LLM providers, embedding models, and retrieval settings.
"""

import os
from typing import Dict, Any, Optional

# ========================
# LLM Provider Settings
# ========================

# Available LLM providers
LLM_PROVIDERS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "available_models": ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini"],
        "default_model": "gpt-4o-mini",
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "available_models": ["gemini-1.5-pro", "gemini-2.5-pro"],
        "default_model": "gemini-2.5-pro",
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "available_models": ["llama-3.3-70b-versatile", "mistral-saba-24b"],
        "default_model": "llama-3.3-70b-versatile",
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "available_models": ["claude-3.7-sonnet", "claude-3.5-sonnet", "claude-3-opus"],
        "default_model": "claude-3.5-sonnet",
    }
}

# Default LLM configuration
LLM_CONFIG = {
    "provider": "openai",  # Change this to switch between providers
    "model": None,  # If None, will use the provider's default model
    "temperature": 0.7,
    "max_tokens": 1000,
}

# ========================
# Embedding Settings
# ========================

# Available embedding providers
EMBEDDING_PROVIDERS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "available_models": ["text-embedding-3-small", "text-embedding-3-large"],
        "default_model": "text-embedding-3-small",
    },
    "sentence_transformers": {
        "api_key_env": None,  # No API key needed for local models
        "available_models": ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        "default_model": "all-MiniLM-L6-v2",
    },
    "cohere": {
        "api_key_env": "COHERE_API_KEY",
        "available_models": ["embed-english-v3.0", "embed-multilingual-v3.0"],
        "default_model": "embed-english-v3.0",
    }
}
# Default embedding configuration
EMBEDDING_CONFIG = {
    "provider": "openai",  # Change this to switch between providers
    "model": None,  # If None, will use the provider's default model
}

# ========================
# Text Processing Settings
# ========================

TEXT_PROCESSING = {
    "chunk_size": 500,
    "chunk_overlap": 50,
}

# ========================
# Vector Store Settings
# ========================

VECTOR_STORE = {
    "type": "faiss",  # Options: "faiss", "chroma", "pinecone"
    "persist_directory": "data/vector_store",
    "search_type": "similarity",
    "search_k": 3,  # Number of documents to retrieve
}

# ========================
# File and Data Settings
# ========================

DATA_CONFIG = {
    "default_documents_dir": "data/documents",
}

# ========================
# Helper Functions
# ========================

#=================================================================================
# get_api_key()
# This function finds the right API key based on which provider you're using.
# Instead of hardcoding API key lookup for each provider throughout your code
# Handles errors gracefully if a key isn't found
# Works for both LLM and embedding providers
#==================================================================================




def get_api_key(provider_type: str, provider_name: str) -> Optional[str]:
    """
    Get the API key for the specified provider from environment variables.
    
    Args:
        provider_type: The type of provider ("llm" or "embedding")
        provider_name: The name of the provider
        
    Returns:
        The API key if found, None otherwise
    """
    providers = LLM_PROVIDERS if provider_type == "llm" else EMBEDDING_PROVIDERS
    
    if provider_name not in providers:
        raise ValueError(f"Unknown {provider_type} provider: {provider_name}")
    
    env_var = providers[provider_name]["api_key_env"]
    if not env_var:
        return None  # No API key needed (e.g., for local models)
    
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(f"API key for {provider_name} not found. Set the {env_var} environment variable.")
    
    return api_key

def get_model_name(provider_type: str, provider_name: str, model_name: Optional[str] = None) -> str:
    """
    Get the model name for the specified provider.
    If model_name is not provided, return the default model for the provider.
    
    Args:
        provider_type: The type of provider ("llm" or "embedding")
        provider_name: The name of the provider
        model_name: The name of the model (optional)
        
    Returns:
        The model name
    """
    providers = LLM_PROVIDERS if provider_type == "llm" else EMBEDDING_PROVIDERS
    
    if provider_name not in providers:
        raise ValueError(f"Unknown {provider_type} provider: {provider_name}")
    
    provider_config = providers[provider_name]
    
    if model_name is None:
        return provider_config["default_model"]
    
    if model_name not in provider_config["available_models"]:
        raise ValueError(f"Unknown model {model_name} for provider {provider_name}")
    
    return model_name

def get_llm_config() -> Dict[str, Any]:
    """
    Get the complete LLM configuration.
    
    Returns:
        A dictionary with the LLM configuration
    """
    config = LLM_CONFIG.copy()
    provider = config["provider"]
    
    # Set the model name if not specified
    if config["model"] is None:
        config["model"] = get_model_name("llm", provider)
    
    # Add the API key
    config["api_key"] = get_api_key("llm", provider)
    
    return config

def get_embedding_config() -> Dict[str, Any]:
    """
    Get the complete embedding configuration.
    
    Returns:
        A dictionary with the embedding configuration
    """
    config = EMBEDDING_CONFIG.copy()
    provider = config["provider"]
    
    # Set the model name if not specified
    if config["model"] is None:
        config["model"] = get_model_name("embedding", provider)
    
    # Add the API key
    config["api_key"] = get_api_key("embedding", provider)
    
    return config