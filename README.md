# Website RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that specifically answers questions about your website content. This chatbot loads content from your website, creates vector embeddings, and provides conversational responses based solely on that content.

## Features

- **Website-Specific**: Answers questions only about your website content
- **Multiple LLM Support**: Works with OpenAI, Google Gemini, Groq, or Anthropic models
- **Vector Storage**: Stores and retrieves vector embeddings for efficient responses
- **Conversation Memory**: Maintains context throughout user interactions
- **Easy Configuration**: Simple configuration without modifying code
- **Interactive Interface**: Friendly, conversational responses

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/website-rag-chatbot.git
   cd website-rag-chatbot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API keys as environment variables:
   ```
   export OPENAI_API_KEY=your_openai_key
   # OR for other providers
   export GEMINI_API_KEY=your_gemini_key
   export GROQ_API_KEY=your_groq_key
   export ANTHROPIC_API_KEY=your_anthropic_key
   ```

## Usage

### Basic Usage

Run the chatbot with your website URL:

```
python main.py --url https://your-website.com --website-name "Your Company Name"
```

### Command Line Options

- `--url`: URL of the website to load content from
- `--use-saved`: Use previously saved vector embeddings instead of loading from URL
- `--vector-path`: Path for saving/loading vector embeddings (default: "data/vector_store")
- `--website-name`: Name of the website for personalization (default: "our website")

### Examples

First-time setup with a website:
```
python main.py --url https://example.com --website-name "Example Corp"
```

Using saved vectors on subsequent runs:
```
python main.py --use-saved --website-name "Example Corp"
```

## Configuration

The chatbot's behavior can be customized by editing the `config.py` file:

### LLM Provider

Change the LLM provider by editing:
```python
LLM_CONFIG = {
    "provider": "openai",  # Change to "gemini", "groq", or "anthropic"
    "model": None,  # Uses default model if None
    "temperature": 0.7,
    "max_tokens": 1000,
}
```

### Embedding Provider

Change the embedding model:
```python
EMBEDDING_CONFIG = {
    "provider": "openai",  # Change to "sentence_transformers" or "cohere"
    "model": None,  # Uses default model if None
}
```

### Text Processing

Adjust chunking parameters:
```python
TEXT_PROCESSING = {
    "chunk_size": 500,
    "chunk_overlap": 50,
}
```

## Project Structure

- `main.py`: Entry point for the application
- `config.py`: Configuration settings
- `document_loader.py`: Loads website content
- `vector_store.py`: Manages vector embeddings
- `llm_service.py`: Interfaces with language models
- `agent.py`: Implements the chatbot logic

## Customization

### Adding New Document Types

Edit `document_loader.py` to add support for additional document types.

### Adding New LLM Providers

Edit `config.py` and `llm_service.py` to add support for additional LLM providers.

## Future Improvements

- Web interface for easier interaction
- Multi-website support
- Fine-tuning options for better responses
- Support for authentication-protected website content
- Query routing for complex websites

## License

[MIT License](LICENSE)