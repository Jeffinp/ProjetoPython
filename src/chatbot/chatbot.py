"""
Classe principal do chatbot Python.
"""

import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import torch

from src.chatbot.config import ChatbotConfig
from src.chatbot.exceptions import ChatbotError, ModelLoadError, ResourceLoadError
from src.chatbot.model import EmbeddingModel
from src.chatbot.nlp import NLPProcessor
from src.chatbot.utils import (
    load_rules,
    is_greeting,
    is_affirmative,
    check_casual_conversation,
    save_conversation,
)


class PythonChatbot:
    """
    Chatbot para responder perguntas sobre Python usando Sentence Transformers.
    Processa entrada em linguagem natural e fornece respostas baseadas em regras
    e similaridade semântica.
    """

    def __init__(self, config: Optional[ChatbotConfig] = None):
        """
        Inicializa o chatbot.

        Args:
            config (Optional[ChatbotConfig]): Configurações personalizadas
        """
        self.config = config or ChatbotConfig()

        # Inicializa o modelo de embeddings
        self.embedding_model = EmbeddingModel(self.config.model_name)

        # Inicializa o processador de NLP
        self.nlp_processor = NLPProcessor()

        self.conversation_history: List[Dict] = []
        self.last_response_type = None
        self.current_topic = None

        # Carrega as regras
        self.rules = load_rules(self.config.rules_file)

        self.casual_patterns = {
            "tudo bem": "como_vai",
            "como vai": "como_vai",
            "como está": "como_vai",
            "beleza": "como_vai",
            "td bem": "como_vai",
        }

        self.rules_embeddings = self.embedding_model.calculate_rule_embeddings(
            self.rules
        )

    def _find_best_response(
        self, question_embedding: torch.Tensor, question: str
    ) -> Tuple[str, float]:
        """Encontra a melhor resposta com base na similaridade."""
        best_similarity = -1
        best_response = None

        for key, rule_embedding in self.rules_embeddings.items():
            similarity = self.embedding_model.calculate_similarity(
                question_embedding, rule_embedding
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_response = key

        if best_similarity < self.config.similarity_threshold:
            casual_response = check_casual_conversation(question, self.casual_patterns)
            if casual_response:
                best_response = casual_response
                best_similarity = 1.0
            else:
                best_response = "fallback"

        return random.choice(self.rules[best_response]), best_similarity

    def _handle_affirmative_response(self) -> Optional[str]:
        """Lida com respostas afirmativas do usuário."""
        if len(self.conversation_history) > 1:
            if self.last_response_type == "pergunta_conceito":
                return self._elaborate_on_last_topic()
        return None

    def _elaborate_on_last_topic(self) -> str:
        """Fornece mais informações sobre o último tópico discutido."""
        if len(self.conversation_history) < 2:
            return "Desculpe, não tenho um tópico anterior para elaborar."

        last_question_type = self.conversation_history[-2].get("type")

        if last_question_type == "pergunta_conceito":
            return self._explain_python_concepts()

        return "Desculpe, não tenho mais informações sobre esse tópico."

    def _explain_python_concepts(self) -> str:
        """Fornece uma explicação detalhada dos conceitos básicos de Python."""
        concept_details = {
            "variaveis": (
                "Variáveis são usadas para armazenar dados. Em Python, você "
                "não precisa declarar explicitamente o tipo de uma variável. "
                "Exemplos: `x = 5`, `nome = 'Alice'`."
            ),
            "listas": (
                "Listas são coleções ordenadas e mutáveis de itens. "
                "Exemplo: `minha_lista = [1, 2, 'python']`."
            ),
            "dicionarios": (
                "Dicionários armazenam pares de chave-valor, permitindo a "
                "recuperação rápida de valores através de chaves. "
                "Exemplo: `d = {'nome': 'Alice', 'idade': 30}`."
            ),
            "tuplas": (
                "Tuplas são como listas, mas são imutáveis, ou seja, não "
                "podem ser alteradas após a criação. "
                "Exemplo: `minha_tupla = (1, 2, 'python')`."
            ),
            "loops": (
                "Loops são usados para executar um bloco de código "
                "repetidamente. Python tem loops `for` e `while`. "
                "Exemplo de loop `for`: `for i in range(5): print(i)`."
            ),
            "funcoes": (
                "Funções são blocos de código reutilizáveis que realizam uma "
                "tarefa específica. "
                "Exemplo: `def saudacao(nome): return 'Olá, ' + nome`."
            ),
            "classes": (
                "Classes são usadas para criar objetos e são a base da "
                "programação orientada a objetos. "
                "Exemplo: `class Cachorro: def __init__(self, nome): "
                "self.nome = nome`."
            ),
            "condicionais": (
                "Condicionais permitem que o programa execute diferentes "
                "blocos de código com base em condições. Exemplo: "
                "`if x > 0: print('Positivo') else: print('Não Positivo')`."
            ),
        }

        concept = random.choice(list(concept_details.keys()))
        return f"Vamos falar sobre {concept}. {concept_details[concept]}"

    def get_response(self, question: str) -> str:
        """
        Gera uma resposta para a pergunta do usuário.

        Args:
            question (str): Pergunta do usuário

        Returns:
            str: Resposta do chatbot
        """
        try:
            if is_greeting(question):
                self.last_response_type = "saudacao"
                self.current_topic = None  # Limpa o tópico em saudações
                return random.choice(self.rules["saudacoes"])

            processed_question = self.nlp_processor.preprocess_question(question)

            if is_affirmative(question) and len(self.conversation_history) > 0:
                response = self._handle_affirmative_response()
                if response:
                    return response

            question_embedding = self.embedding_model.encode_text(processed_question)

            response, similarity = self._find_best_response(
                question_embedding, question
            )

            # Atualizar o tópico com base na resposta
            if (
                similarity > self.config.similarity_threshold
            ):  # Só atualiza se a resposta for considerada boa
                if "como aprender python" in question.lower():
                    self.current_topic = "aprender_python"
                elif "python" in question.lower():
                    self.current_topic = "python"
                else:
                    self.current_topic = None

            # Usar o tópico atual para refinar a resposta, se necessário
            if (
                self.current_topic == "aprender_python"
                and "como aprender" in question.lower()
            ):
                response = random.choice(self.rules["como aprender python?"])
                self.last_response_type = "pergunta_conceito"

            elif "conceit" in processed_question:
                self.last_response_type = "pergunta_conceito"
            else:
                self.last_response_type = "geral"

            self.conversation_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "processed_question": processed_question,
                    "response": response,
                    "similarity_score": similarity,
                    "type": self.last_response_type,
                    "topic": self.current_topic,
                }
            )

            return response

        except Exception as e:
            logging.exception("Erro ao gerar resposta: %s", str(e))
            return (
                "Desculpe, ocorreu um erro ao processar sua pergunta. "
                "Pode tentar novamente?"
            )

    def save_conversation(self) -> None:
        """
        Salva o histórico da conversa em arquivo JSON.
        """
        save_conversation(self.conversation_history, self.config.history_file)

    def run(self) -> None:
        """
        Executa o loop interativo do chatbot.
        """
        print(
            "Olá! Eu sou um chatbot sobre Python. Pergunte-me qualquer coisa!"
            " (Digite 'sair' para encerrar)"
        )

        try:
            while True:
                question = input("Você: ").strip()

                if question.lower() == "sair":
                    print("Chatbot:", random.choice(self.rules["despedidas"]))
                    self.save_conversation()
                    break

                response = self.get_response(question)
                print("Chatbot:", response)

        except KeyboardInterrupt:
            print("\nEncerrando o chatbot...")
            self.save_conversation()
        except ChatbotError as e:
            logging.error("Erro do chatbot: %s", str(e))
            print(f"Erro: {str(e)}")
            self.save_conversation()
