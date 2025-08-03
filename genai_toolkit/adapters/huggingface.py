import logging
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from ..base_model import Model
from ..errors import InferenceError
from ..security import SecurityManager

logger = logging.getLogger(__name__)

class HuggingFaceAdapter(Model):
    """LangChain-based adapter for Hugging Face models with enterprise key security."""
    def __init__(self, enterprise_app_key: str, model_name: str = "gpt2", cache_ttl: int = 3600, redis_config: dict = None):
        super().__init__(cache_ttl, redis_config)
        try:
            security = SecurityManager(enterprise_app_key)
            api_key = security.get_decrypted_api_key("huggingface")
            self.model = HuggingFacePipeline.from_model_id(
                model_id=model_name,
                task="text-generation",
                pipeline_kwargs={"max_length": 100}
            )
            self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            logger.info("HuggingFaceAdapter initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize HuggingFaceAdapter: %s", str(e))
            raise InferenceError(f"Failed to initialize Hugging Face model: {str(e)}")

    def generate(self, prompt: str, **kwargs) -> str:
        cache_key = self._cache_key("generate", prompt, **kwargs)
        if cached_result := self.cache.get(cache_key):
            logger.info("Returning cached result for prompt: %s", prompt[:50])
            return cached_result

        try:
            result = self.model.invoke(prompt, **kwargs)
            text = result.strip()
            self.cache.set(cache_key, text)
            logger.info("Generated response for prompt: %s", prompt[:50])
            return text
        except Exception as e:
            logger.error("Hugging Face generation failed: %s", str(e))
            raise InferenceError(f"Hugging Face generation failed: {str(e)}")

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
            logger.error("Hugging Face embedding failed: %s", str(e))
            raise InferenceError(f"Hugging Face embedding failed: {str(e)}")