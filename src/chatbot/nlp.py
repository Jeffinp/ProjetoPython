"""
M�dulo de processamento de linguagem natural para o chatbot.
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
            # Baixar recursos do NLTK independentemente de j� estarem presentes
            nltk.download("stopwords", quiet=True)
            nltk.download("rslp", quiet=True)

            # Verificar se os recursos foram baixados corretamente
            try:
                self.stopwords = set(nltk.corpus.stopwords.words("portuguese"))
                self.stemmer = RSLPStemmer()
            except LookupError:
                # Se falhar, tente novamente com quiet=False para mostrar mais informa��es
                print("Tentando baixar recursos NLTK novamente...")
                nltk.download("stopwords", quiet=False)
                nltk.download("rslp", quiet=False)
                self.stopwords = set(nltk.corpus.stopwords.words("portuguese"))
                self.stemmer = RSLPStemmer()
        except Exception as e:
            raise ResourceLoadError(f"Erro ao baixar recursos NLTK: {e}") from e

    def remove_acentos(self, texto: str) -> str:
        """Remove acentos de uma string."""
        return unidecode(texto)

    def expandir_abreviacoes(self, texto: str) -> str:
        """Expande abrevia��es comuns."""
        abreviacoes = {
            "vc": "voc�",
            "pq": "porque",
            "oq": "o que",
            "eh": "�",
            "q": "que",
            "tb": "tamb�m",
            "tbm": "tamb�m",
            "blz": "beleza",
            "cmg": "comigo",
            "sao": "s�o",
            "obg": "obrigado",
            "mt": "muito",
            "msg": "mensagem",
            "agr": "agora",
            "vdd": "verdade",
            "mto": "muito",
            "to": "estou",
            "ta": "est�",
            "n": "n�o",
            "nao": "n�o",
            "td": "tudo",
            "t�": "est�",
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
        """Aplica stemming �s palavras de uma string."""
        palavras = [self.stemmer.stem(palavra) for palavra in texto.split()]
        return " ".join(palavras)

    def preprocess_question(self, question: str) -> str:
        """Pr�-processa a pergunta."""
        question = self.remove_acentos(question.lower())
        question = self.expandir_abreviacoes(question)
        question = self.remover_stopwords(question)
        question = self.stemming(question)
        return question
