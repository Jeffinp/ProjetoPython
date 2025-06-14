"""
Módulo de processamento de linguagem natural para o chatbot.
"""

import nltk
from nltk.stem import RSLPStemmer
from unidecode import unidecode

from src.chatbot.exceptions import ResourceLoadError


class NLPProcessor:
    """
    Classe para processamento de linguagem natural.
    """
    
    def __init__(self):
        """Inicializa o processador de NLP."""
        self._initialize_nlp_resources()
        
    def _initialize_nlp_resources(self) -> None:
        """
        Inicializa recursos de NLP (NLTK).

        Raises:
            ResourceLoadError: Se houver erro ao carregar recursos
        """
        try:
            # Baixar recursos do NLTK independentemente de já estarem presentes
            nltk.download("stopwords", quiet=False)
            nltk.download("rslp", quiet=False)
            
            # Verificar se os recursos foram baixados corretamente
            if not nltk.data.find("corpora/stopwords"):
                raise LookupError("Recurso 'stopwords' não foi baixado corretamente")
            if not nltk.data.find("stemmers/rslp"):
                raise LookupError("Recurso 'rslp' não foi baixado corretamente")
                
            self.stopwords = set(nltk.corpus.stopwords.words("portuguese"))
            self.stemmer = RSLPStemmer()
        except Exception as e:
            raise ResourceLoadError(f"Erro ao baixar recursos NLTK: {e}") from e

    def remove_acentos(self, texto: str) -> str:
        """Remove acentos de uma string."""
        return unidecode(texto)

    def expandir_abreviacoes(self, texto: str) -> str:
        """Expande abreviações comuns."""
        abreviacoes = {
            "vc": "você",
            "pq": "porque",
            "oq": "o que",
            "eh": "é",
            "q": "que",
            "tb": "também",
            "tbm": "também",
            "blz": "beleza",
            "cmg": "comigo",
            "sao": "são",
            "obg": "obrigado",
            "mt": "muito",
            "msg": "mensagem",
            "agr": "agora",
            "vdd": "verdade",
            "mto": "muito",
            "to": "estou",
            "ta": "está",
            "n": "não",
            "nao": "não",
            "td": "tudo",
            "tá": "está",
            "pra": "para",
            "pro": "para o",
            "c/": "com",
        }
        palavras = texto.split()
        palavras_expandidas = [
            abreviacoes.get(palavra, palavra) for palavra in palavras
        ]
        return " ".join(palavras_expandidas)

    def remover_stopwords(self, texto: str) -> str:
        """Remove stopwords de uma string."""
        palavras = [
            palavra for palavra in texto.split() if palavra not in self.stopwords
        ]
        return " ".join(palavras)

    def stemming(self, texto: str) -> str:
        """Aplica stemming às palavras de uma string."""
        palavras = [self.stemmer.stem(palavra) for palavra in texto.split()]
        return " ".join(palavras)

    def preprocess_question(self, question: str) -> str:
        """Pré-processa a pergunta."""
        question = self.remove_acentos(question.lower())
        question = self.expandir_abreviacoes(question)
        question = self.remover_stopwords(question)
        question = self.stemming(question)
        return question
