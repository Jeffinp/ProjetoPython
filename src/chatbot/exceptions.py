"""
Módulo de exceções para o chatbot.
"""


class ChatbotError(Exception):
    """Base exception class for chatbot-specific errors"""


class ModelLoadError(ChatbotError):
    """Raised when there are issues loading the model"""


class ResourceLoadError(ChatbotError):
    """Raised when there are issues loading resources"""
