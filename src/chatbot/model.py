"""
Módulo do modelo de embeddings para o chatbot.
"""

import logging
import torch
from sentence_transformers import SentenceTransformer, util
from typing import Dict

from src.chatbot.exceptions import ModelLoadError


class EmbeddingModel:
    """
    Classe para gerenciar o modelo de embeddings do chatbot.
    """

    def __init__(self, model_name: str):
        """
        Inicializa o modelo de embeddings.

        Args:
            model_name (str): Nome do modelo a ser carregado
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Usando device: %s", self.device)
        self.model = self._load_model()

    def _load_model(self) -> SentenceTransformer:
        """
        Carrega o modelo de embeddings.

        Returns:
            SentenceTransformer: Modelo carregado

        Raises:
            ModelLoadError: Se houver erro ao carregar o modelo
        """
        try:
            model = SentenceTransformer(self.model_name, device=self.device)
            logging.info("Modelo '%s' carregado com sucesso", self.model_name)
            return model
        except Exception as e:
            logging.error("Erro ao carregar modelo: %s", str(e))
            raise ModelLoadError(f"Erro ao carregar modelo: {e}") from e

    def calculate_rule_embeddings(self, rules: Dict) -> Dict[str, torch.Tensor]:
        """
        Calcula e armazena em cache os embeddings das regras.

        Args:
            rules (Dict): Dicionário de regras do chatbot

        Returns:
            Dict[str, torch.Tensor]: Dicionário de embeddings
        """
        embeddings = {}
        for key in rules:
            if key != "repetir":
                embeddings[key] = self.model.encode(
                    [key], convert_to_tensor=True, show_progress_bar=False
                )
        return embeddings

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Codifica um texto para um tensor de embedding.

        Args:
            text (str): Texto a ser codificado

        Returns:
            torch.Tensor: Tensor de embedding
        """
        return self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)

    def calculate_similarity(
        self, embedding1: torch.Tensor, embedding2: torch.Tensor
    ) -> float:
        """
        Calcula a similaridade entre dois embeddings.

        Args:
            embedding1 (torch.Tensor): Primeiro embedding
            embedding2 (torch.Tensor): Segundo embedding

        Returns:
            float: Valor de similaridade
        """
        return util.pytorch_cos_sim(embedding1, embedding2)[0].max().item()
