# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that specifically answers questions about your company content. This chatbot loads content from your website, text files, or PDF documents, creates vector embeddings, and provides conversational responses based solely on that content.

## Features

- **Content-Specific**: Answers questions only about your provided content
- **Multiple Data Sources**: Support for websites, text files, and PDFs
- **Multiple LLM Support**: Works with OpenAI, Google Gemini, Groq, or Anthropic models
- **Vector Storage**: Stores and retrieves vector embeddings for efficient responses
- **Conversation Memory**: Maintains context throughout user interactions
- **Easy Configuration**: Simple configuration without modifying code
- **Interactive Interface**: Friendly, conversational responses

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API keys as environment variables:

   **On Linux/Mac:**
   ```
   export OPENAI_API_KEY=your_openai_key
   ```

   **On Windows (Command Prompt):**
   ```
   set OPENAI_API_KEY=your_openai_key
   ```

   **On Windows (PowerShell):**
   ```
   $env:OPENAI_API_KEY = "your_openai_key"
   ```

   For other providers, use:
   - GEMINI_API_KEY
   - GROQ_API_KEY
   - ANTHROPIC_API_KEY

## Usage

### Basic Usage

Run the chatbot with your chosen content source:

```
# With a website
python main.py --url https://your-website.com --website-name "Your Company Name"

# With a text file
python main.py --text-file data/company_info.txt --website-name "Your Company Name"
```

### Command Line Options

- `--url`: URL of the website to load content from
- `--text-file`: Path to a text file to load content from
- `--use-saved`: Use previously saved vector embeddings instead of loading content again
- `--vector-path`: Path for saving/loading vector embeddings (default: "data/vector_store")
- `--website-name`: Name of your company for personalization (default: "our website")

### Examples

First-time setup with a website:
```
python main.py --url https://example.com --website-name "Example Corp"
```

First-time setup with a text file:
```
python main.py --text-file data/company_info.txt --website-name "Example Corp"
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
- `document_loader.py`: Loads content from various sources
- `vector_store.py`: Manages vector embeddings
- `llm_service.py`: Interfaces with language models
- `agent.py`: Implements the chatbot logic

## Troubleshooting

### API Key Issues

If you see an error about missing API keys, ensure you've set them correctly:
- Check that you've set the environment variable without quotes (in Command Prompt)
- For PowerShell, quotes are required: `$env:OPENAI_API_KEY = "your-key-here"`
- Verify there are no spaces around the equals sign

### Chatbot Not Starting

If the chatbot fails to start:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that your content source (URL or file) is accessible
- Look for detailed error messages in the console output

## Customization

### Adding New Document Types

Edit `document_loader.py` to add support for additional document types.

### Adding New LLM Providers

Edit `config.py` and `llm_service.py` to add support for additional LLM providers.

## Future Improvements

- Web interface for easier interaction
- Multi-content source support in a single session
- Fine-tuning options for better responses
- Support for authentication-protected content
- Query routing for complex information structures

## License

[MIT License](LICENSE)