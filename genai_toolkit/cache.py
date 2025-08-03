import redis
import os
import pickle
import logging
from .errors import InferenceError

logger = logging.getLogger(__name__)

class ResponseCache:
    """Distributed cache for model responses using Redis."""
    def __init__(self, ttl: int = 3600, host: str = None, port: int = None, db: int = 0, password: str = None, ssl: bool = False):
        try:
            self.ttl = ttl
            self.redis = redis.Redis(
                host=host or os.getenv("REDIS_HOST", "localhost"),
                port=port or int(os.getenv("REDIS_PORT", 6379)),
                db=db,
                password=password or os.getenv("REDIS_PASSWORD"),
                ssl=ssl or os.getenv("REDIS_SSL", "False").lower() == "true",
                decode_responses=False
            )
            # Test connection
            self.redis.ping()
            logger.info("Connected to Redis at %s:%s", self.redis.connection_pool.connection_kwargs["host"], self.redis.connection_pool.connection_kwargs["port"])
        except redis.RedisError as e:
            logger.error("Failed to connect to Redis: %s", str(e))
            raise InferenceError(f"Failed to connect to Redis: {str(e)}")

    def get(self, key: str):
        """Retrieve a cached value by key."""
        try:
            value = self.redis.get(key)
            if value is not None:
                return pickle.loads(value)
            return None
        except redis.RedisError as e:
            logger.error("Redis get failed for key %s: %s", key, str(e))
            return None

    def set(self, key: str, value):
        """Set a cached value with TTL."""
        try:
            serialized_value = pickle.dumps(value)
            self.redis.setex(key, self.ttl, serialized_value)
            logger.debug("Cached value for key %s with TTL %s", key, self.ttl)
        except redis.RedisError as e:
            logger.error("Redis set failed for key %s: %s", key, str(e))