import os
import json
import logging
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from .errors import SecurityError

logger = logging.getLogger(__name__)

class SecurityManager:
    """Manages encryption and decryption of API keys using enterprise app key."""
    def __init__(self, enterprise_app_key: str, config_path: str = "config.json"):
        if not self._validate_key(enterprise_app_key):
            raise SecurityError("Invalid enterprise app key: must be 32 bytes")
        self.key = enterprise_app_key.encode('utf-8')
        self.config_path = config_path
        logger.info("SecurityManager initialized with config path: %s", config_path)

    def _validate_key(self, key: str) -> bool:
        """Validate the enterprise app key length (32 bytes for AES-256)."""
        return len(key.encode('utf-8')) == 32

    def decrypt_api_key(self, encrypted_key: bytes, nonce: bytes) -> str:
        """Decrypt an API key using the enterprise app key."""
        try:
            aesgcm = AESGCM(self.key)
            decrypted_key = aesgcm.decrypt(nonce, encrypted_key, None)
            logger.info("Successfully decrypted API key")
            return decrypted_key.decode('utf-8')
        except Exception as e:
            logger.error("Decryption failed: %s", str(e))
            raise SecurityError(f"Failed to decrypt API key: {str(e)}")

    def load_config(self) -> dict:
        """Load encrypted API keys from config file."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info("Loaded config from %s", self.config_path)
            return config
        except Exception as e:
            logger.error("Failed to load config: %s", str(e))
            raise SecurityError(f"Failed to load config: {str(e)}")

    def get_decrypted_api_key(self, provider: str) -> str:
        """Get decrypted API key for a specific provider."""
        config = self.load_config()
        if provider not in config:
            logger.error("Provider %s not found in config", provider)
            raise SecurityError(f"Provider {provider} not found in config")
        encrypted_key = bytes.fromhex(config[provider]['encrypted_key'])
        nonce = bytes.fromhex(config[provider]['nonce'])
        return self.decrypt_api_key(encrypted_key, nonce)