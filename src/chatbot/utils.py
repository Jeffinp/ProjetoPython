"""
Módulo com funções utilitárias para o chatbot.
"""

import json
import logging
import random
from typing import Dict, List, Optional, Tuple

from src.chatbot.exceptions import ResourceLoadError


def load_rules(file_path: str) -> Dict:
    """
    Carrega as regras do chatbot de um arquivo JSON.

    Args:
        file_path (str): Caminho para o arquivo de regras

    Returns:
        Dict: Dicionário com as regras

    Raises:
        ResourceLoadError: Se houver erro ao carregar as regras
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
        logging.info("Regras carregadas com sucesso")
        return rules
    except FileNotFoundError as exc:
        msg = f"Arquivo de regras não encontrado: {file_path}"
        logging.error(msg)
        raise ResourceLoadError(msg) from exc
    except json.JSONDecodeError as e:
        msg = f"Erro ao decodificar JSON do arquivo de regras: {e}"
        logging.error(msg)
        raise ResourceLoadError(msg) from e


def is_greeting(text: str) -> bool:
    """
    Verifica se o texto é uma saudação.

    Args:
        text (str): Texto de entrada

    Returns:
        bool: True se for uma saudação
    """
    greetings = {
        "olá",
        "oi",
        "bom dia",
        "boa tarde",
        "boa noite",
        "e aí",
        "fala aí",
        "opa",
        "eae",
    }
    text = text.lower().strip()
    return any(text.startswith(greeting) for greeting in greetings)


def is_affirmative(text: str) -> bool:
    """
    Verifica se o texto é uma resposta afirmativa.

    Args:
        text (str): Texto de entrada

    Returns:
        bool: True se for uma resposta afirmativa
    """
    affirmative_words = {
        "sim",
        "quero",
        "claro",
        "com certeza",
        "pode ser",
        "afirmativo",
        "ok",
        "beleza",
        "isso",
        "exato",
        "certo",
    }
    text = text.lower().strip()
    return text in affirmative_words


def check_casual_conversation(
    question: str, casual_patterns: Dict[str, str]
) -> Optional[str]:
    """
    Verifica se o texto é uma conversa casual.

    Args:
        question (str): Texto de entrada
        casual_patterns (Dict[str, str]): Padrões de conversa casual

    Returns:
        Optional[str]: Tipo de resposta casual ou None
    """
    text = question.lower().strip()
    return next(
        (
            response_type
            for pattern, response_type in casual_patterns.items()
            if pattern in text
        ),
        None,
    )


def save_conversation(history: List[Dict], file_path: str) -> None:
    """
    Salva o histórico da conversa em arquivo JSON.

    Args:
        history (List[Dict]): Histórico da conversa
        file_path (str): Caminho para o arquivo de histórico
    """
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=4)
        logging.info("Histórico da conversa salvo com sucesso")
    except IOError as e:
        logging.error("Erro ao salvar histórico: %s", str(e))
        print("Aviso: Não foi possível salvar o histórico da conversa")
