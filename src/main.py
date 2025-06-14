"""
Ponto de entrada principal para o chatbot Python.
"""

import logging
from src.chatbot.config import ChatbotConfig
from src.chatbot.chatbot import PythonChatbot
from src.chatbot.exceptions import ChatbotError


def main():
    """Função principal para executar o chatbot."""
    try:
        config = ChatbotConfig()
        chatbot = PythonChatbot(config)
        chatbot.run()
    except ChatbotError as e:
        logging.error("Erro fatal: %s", str(e))
        print(f"Erro crítico: {str(e)}")
    except Exception as e:
        logging.error("Erro não esperado: %s", str(e))
        print("Ocorreu um erro inesperado. O chatbot será encerrado.")


if __name__ == "__main__":
    main()
