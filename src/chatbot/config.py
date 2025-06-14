"""
Módulo de configuração para o chatbot.
"""

import logging
from dataclasses import dataclass


@dataclass
class ChatbotConfig:
    """Configuration class for the Python chatbot"""

    model_name: str = "rufimelo/bert-large-portuguese-cased-sts"
    log_file: str = "logs/chatbot.log"
    history_file: str = "src/data/conversation_history.json"
    rules_file: str = "src/data/rules.json"
    similarity_threshold: float = 0.5


# Configure logging globally
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/chatbot.log"), logging.StreamHandler()],
)
