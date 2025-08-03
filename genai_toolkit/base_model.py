from abc import ABC, abstractmethod
from .cache import ResponseCache
from .errors import InferenceError

class Model(ABC):
    """Abstract base class for GenAI model inference with LangChain."""
    def __init__(self, cache_ttl=3600, redis_config=None):
        self.cache = ResponseCache(ttl=cache_ttl, **(redis_config or {}))

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on a prompt."""
        pass

    @abstractmethod
    def summarize(self, text: str, **kwargs) -> str:
        """Summarize input text."""
        pass

    @abstractmethod
    def embed(self, text: str, **kwargs) -> list:
        """Generate embeddings for input text."""
        pass

    def _cache_key(self, method: str, *args, **kwargs) -> str:
        """Generate a cache key from method name and arguments."""
        return f"{method}:{str(args)}:{str(kwargs)}"