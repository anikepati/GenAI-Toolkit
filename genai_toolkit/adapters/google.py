import logging
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ..base_model import Model
from ..errors import InferenceError, RateLimitError, ConnectionError
from ..security import SecurityManager

logger = logging.getLogger(__name__)

class GoogleGeminiAdapter(Model):
    """LangChain-based adapter for Google Gemini models with enterprise key security."""
    def __init__(self, enterprise_app_key: str, model: str = "gemini-1.5-pro", cache_ttl: int = 3600, redis_config: dict = None):
        super().__init__(cache_ttl, redis_config)
        try:
            security = SecurityManager(enterprise_app_key)
            api_key = security.get_decrypted_api_key("google")
            self.model = ChatGoogleGenerativeAI(google_api_key=api_key, model=model)
            self.embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/text-embedding-004")
            logger.info("GoogleGeminiAdapter initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize GoogleGeminiAdapter: %s", str(e))
            raise ConnectionError(f"Failed to initialize Google Gemini model: {str(e)}")

    def generate(self, prompt: str, **kwargs) -> str:
        cache_key = self._cache_key("generate", prompt, **kwargs)
        if cached_result := self.cache.get(cache_key):
            logger.info("Returning cached result for prompt: %s", prompt[:50])
            return cached_result

        try:
            result = self.model.invoke(prompt, **kwargs)
            text = result.content
            self.cache.set(cache_key, text)
            logger.info("Generated response for prompt: %s", prompt[:50])
            return text
        except Exception as e:
            logger.error("Google Gemini generation failed: %s", str(e))
            if "rate limit" in str(e).lower():
                raise RateLimitError("Google Gemini API rate limit exceeded")
            raise InferenceError(f"Google Gemini generation failed: {str(e)}")

    def summarize(self, text: str, **kwargs) -> str:
        summary_prompt = f"Summarize the following text in 100 words or less:\n\n{text}"
        return self.generate(summary_prompt, **kwargs)

    def embed(self, text: str, **kwargs) -> list:
        cache_key = self._cache_key("embed", text, **kwargs)
        if cached_result := self.cache.get(cache_key):
            logger.info("Returning cached embedding for text: %s", text[:50])
            return cached_result

        try:
            embedding = self.embedding_model.embed_query(text)
            self.cache.set(cache_key, embedding)
            logger.info("Generated embedding for text: %s", text[:50])
            return embedding
        except Exception as e:
            logger.error("Google Gemini embedding failed: %s", str(e))
            raise InferenceError(f"Google Gemini embedding failed: {str(e)}")