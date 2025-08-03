# GenAI Toolkit

A secure, LangChain-based Python package for enterprise-grade Generative AI and Retrieval-Augmented Generation (RAG) applications. The `genai_toolkit` provides a unified interface for prompt engineering, model inference, evaluation, and conversation history summarization, with distributed caching using Redis and robust security enforced through an enterprise application key.

## Features

- **Prompt Engineering**: Manage and format prompts using LangChain's `ChatPromptTemplate` for consistent and reusable prompt creation.
- **Model Inference**: Unified interface for interacting with OpenAI, Google Gemini, and Hugging Face models, supporting text generation, summarization, and embeddings.
- **Evaluation Metrics**: Assess RAG system performance with BLEU, ROUGE, context relevance, and answer relevance metrics.
- **History Summarization**: Summarize conversation history to optimize token usage, reducing costs and improving performance in RAG applications.
- **Distributed Caching**: Uses Redis for scalable, distributed caching of model responses, ideal for enterprise environments.
- **Enterprise Security**: Requires a 32-byte enterprise app key to decrypt LLM API keys, ensuring only authorized applications can access LLM functionality.
- **Error Handling**: Robust handling of rate limits, connection issues, and security errors with detailed logging for auditing.

## Installation

Install the package from GitHub using `pip`:

```bash
pip install git+https://github.com/yourusername/genai_toolkit.git
```

For private repositories, use a GitHub Personal Access Token (PAT) with `repo` scope:

```bash
pip install git+https://<USERNAME>:<PERSONAL_ACCESS_TOKEN>@github.com/yourusername/genai_toolkit.git
```

## Setup

1. **Set Up Redis**:

   - Install and run a Redis server (e.g., locally, on AWS ElastiCache, or Redis Enterprise).
   - Configure Redis connection parameters via environment variables:

     ```bash
     export REDIS_HOST=localhost
     export REDIS_PORT=6379
     export REDIS_PASSWORD=your-redis-password  # Optional
     export REDIS_SSL=false                    # Set to true for SSL
     ```
   - Alternatively, pass parameters directly when initializing models (see Usage).

2. **Generate an Enterprise App Key**:

   - Your team provides a 32-byte enterprise app key (e.g., generate using `openssl rand -base64 32`).
   - This key is required to decrypt LLM API keys and access the package's functionality.

3. **Encrypt LLM API Keys**:

   - Use the provided `encrypt_keys.py` script to encrypt API keys for OpenAI, Google Gemini, and Hugging Face.
   - Example:

     ```python
     from encrypt_keys import generate_config
     enterprise_app_key = "your-32-byte-enterprise-app-key"
     api_keys = {
         "openai": "your-openai-api-key",
         "google": "your-google-api-key",
         "huggingface": "your-huggingface-api-key"
     }
     generate_config(enterprise_app_key, api_keys, output_path="config.json")
     ```
   - This generates a `config.json` file with encrypted keys and nonces.

4. **Place** `config.json`:

   - Store `config.json` in the root directory of your application or specify its path when initializing models.

5. **Install Dependencies**:

   - Ensure all dependencies are installed (see `requirements.txt`):

     ```bash
     pip install -r requirements.txt
     ```

## Requirements

- Python 3.8+
- Redis server (version 6.0 or higher recommended)
- Dependencies:
  - `langchain>=0.3.0`
  - `langchain-openai>=0.2.0`
  - `langchain-google-genai>=1.0.10`
  - `langchain-huggingface>=0.3.0`
  - `redis>=5.0.8`
  - `nltk>=3.8.1`
  - `rouge-score>=0.1.2`
  - `numpy>=1.26.4`
  - `cryptography>=43.0.1`

## Usage

The `genai_toolkit` package supports enterprise RAG applications with prompt management, model inference, evaluation, and history summarization. Below is an example of using the package in a RAG system with Redis caching:

```python
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from genai_toolkit import OpenAIAdapter, PromptManager, Evaluator, HistorySummarizer

# Initialize model with enterprise app key and Redis configuration
enterprise_app_key = "your-32-byte-enterprise-app-key"
redis_config = {
    "host": "localhost",
    "port": 6379,
    "password": None,  # Set if required
    "ssl": False
}
model = OpenAIAdapter(enterprise_app_key=enterprise_app_key, model="gpt-4", redis_config=redis_config)

# Set up prompt manager
prompt_manager = PromptManager(system_prompt="You are a RAG assistant.")
prompt_manager.create_template("rag", "Context: {context}\nQuestion: {question}", ["context", "question"])

# Set up history summarizer
summarizer = HistorySummarizer(model, max_summary_length=50)

# Set up evaluator
evaluator = Evaluator(model)

# Mock vector store for RAG
docs = [Document(page_content="AI is a field of computer science focusing on intelligent systems.")]
vectorstore = FAISS.from_documents(docs, model.embedding_model)
retriever = vectorstore.as_retriever()

# Create RAG chain
rag_prompt = prompt_manager.get_template("rag")
rag_chain = RetrievalQA.from_chain_type(
    llm=model.model,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": rag_prompt}
)

# Conversation history
history = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is the simulation of human intelligence in machines."}
]

# Summarize history to reduce tokens
summary = summarizer.summarize(history)
print(f"History Summary: {summary}")

# Run RAG query with summarized history
query = f"Summary of previous conversation: {summary}\nCurrent question: What is AI?"
result = rag_chain.run(query)
print(f"RAG Response: {result}")

# Evaluate RAG performance
metrics = evaluator.evaluate_rag(
    query=query,
    context=docs[0].page_content,
    generated_answer=result,
    reference_answer="AI enables machines to mimic human intelligence."
)
print(f"RAG Metrics: {metrics}")
```

## Security

The package enforces enterprise-grade security:

- **Enterprise App Key**: A 32-byte key is required to decrypt LLM API keys, ensuring only authorized applications can use the package.
- **AES-256-GCM Encryption**: LLM API keys are stored encrypted in `config.json` and decrypted at runtime.
- **Logging**: All key validation, decryption attempts, and LLM calls are logged for auditing.
- **Middle Layer**: The package acts as a secure intermediary, preventing direct access to LLM APIs by custom applications.

## Enterprise Features

- **Token Optimization**: The `HistorySummarizer` reduces token usage by summarizing conversation history, critical for cost-effective RAG systems.
- **RAG Evaluation**: The `Evaluator` provides BLEU, ROUGE, context relevance, and answer relevance metrics to assess RAG performance.
- **Distributed Caching**: Uses Redis for scalable caching of model responses, suitable for multi-instance deployments.
- **Extensibility**: Easily add new model adapters (e.g., Anthropic, Cohere) or evaluation metrics.

## Extending for Enterprise Use

To enhance the package for production:

- **Async Support**: Add async methods to adapters for high-throughput scenarios:

  ```python
  async def generate(self, prompt: str, **kwargs) -> str:
      cache_key = self._cache_key("generate", prompt, **kwargs)
      if cached_result := self.cache.get(cache_key):
          return cached_result
      result = await self.model.ainvoke(prompt, **kwargs)
      text = result.content
      self.cache.set(cache_key, text)
      return text
  ```
- **Monitoring**: Integrate with Prometheus or Datadog for metrics on API usage and latency.
- **CI/CD**: Use GitHub Actions for automated testing and deployment:

  ```yaml
  name: CI
  on: [push]
  jobs:
    test:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: '3.8'
        - run: pip install -r requirements.txt
        - run: python -m unittest discover tests
  ```
- **Key Management**: Integrate with AWS KMS or HashiCorp Vault for secure enterprise app key rotation.
- **Redis Deployment**: Use managed Redis services (e.g., AWS ElastiCache, Redis Enterprise) with SSL and authentication for production.

## Testing

Run unit tests to verify functionality:

```bash
python -m unittest discover tests
```

## Contributing

Contributions are welcome! Please submit pull requests or issues to the GitHub repository.

## License

MIT License. See LICENSE for details.
