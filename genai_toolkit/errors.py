class InferenceError(Exception):
    """Base exception for inference errors."""
    pass

class RateLimitError(InferenceError):
    """Exception for API rate limit errors."""
    pass

class ConnectionError(InferenceError):
    """Exception for API connection errors."""
    pass

class SecurityError(InferenceError):
    """Exception for security-related errors."""
    pass