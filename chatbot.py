import logging
import random
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import torch
from sentence_transformers import SentenceTransformer, util
from unidecode import unidecode
import nltk
from nltk.stem import RSLPStemmer
from dataclasses import dataclass

# Exceptions
class ChatbotError(Exception):
    """Base exception class for chatbot-specific errors"""
    pass

class ModelLoadError(ChatbotError):
    """Raised when there are issues loading the model"""
    pass

class ResourceLoadError(ChatbotError):
    """Raised when there are issues loading resources"""
    pass

# Config
@dataclass
class ChatbotConfig:
    model_name: str = 'neuralmind/bert-base-portuguese-cased'
    log_file: str = 'chatbot.log'
    history_file: str = 'conversation_history.json'
    rules_file: str = 'rules.json'
    similarity_threshold: float = 0.5

# Configure logging globally
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)

class PythonChatbot:
    """
    Chatbot para responder perguntas sobre Python usando Sentence Transformers.
    """
    def __init__(self, config: Optional[ChatbotConfig] = None):
        """
        Inicializa o chatbot.

        Args:
            config (Optional[ChatbotConfig]): Configurações personalizadas
        """
        self.config = config or ChatbotConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Usando device: {self.device}")

        self.model = self._load_model()
        self.conversation_history: List[Dict] = []
        self.last_response_type = None  # Adicionado para rastrear o tipo da última resposta

        self._initialize_nlp_resources()
        self._load_rules()

    def _load_model(self) -> SentenceTransformer:
        """
        Carrega o modelo de embeddings.

        Returns:
            SentenceTransformer: Modelo carregado

        Raises:
            ModelLoadError: Se houver erro ao carregar o modelo
        """
        try:
            model = SentenceTransformer(self.config.model_name, device=self.device)
            logging.info(f"Modelo '{self.config.model_name}' carregado com sucesso")
            return model
        except Exception as e:
            logging.error(f"Erro ao carregar modelo: {e}")
            raise ModelLoadError(f"Erro ao carregar modelo: {e}")

    def _initialize_nlp_resources(self) -> None:
        """
        Inicializa recursos de NLP (NLTK).

        Raises:
            ResourceLoadError: Se houver erro ao carregar recursos
        """
        try:
            if not nltk.data.find('corpora/stopwords'):
                nltk.download('stopwords', quiet=True)
            if not nltk.data.find('stemmers/rslp'):
                nltk.download('rslp', quiet=True)
            self.stopwords = set(nltk.corpus.stopwords.words('portuguese'))
            self.stemmer = RSLPStemmer()
        except LookupError as e:
            logging.error(f"Erro ao baixar recursos NLTK: {e}")
            raise ResourceLoadError(f"Erro ao baixar recursos NLTK: {e}")

    def _load_rules(self) -> None:
        """
        Carrega as regras do chatbot e calcula os embeddings.

        Raises:
            ResourceLoadError: Se houver erro ao carregar as regras
        """
        try:
            with open(self.config.rules_file, "r", encoding="utf-8") as f:
                self.rules = json.load(f)
            logging.info("Regras carregadas com sucesso")
        except FileNotFoundError:
            logging.error(f"Arquivo de regras não encontrado: {self.config.rules_file}")
            raise ResourceLoadError(f"Arquivo de regras não encontrado: {self.config.rules_file}")
        except json.JSONDecodeError as e:
            logging.error(f"Erro ao decodificar JSON do arquivo de regras: {e}")
            raise ResourceLoadError(f"Erro ao decodificar JSON do arquivo de regras: {e}")

        self.casual_patterns = {
            "tudo bem": "como_vai",
            "como vai": "como_vai",
            "como está": "como_vai",
            "beleza": "como_vai",
            "td bem": "como_vai",
        }

        self.rules_embeddings = self._calculate_rule_embeddings()

    def _calculate_rule_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Calcula e armazena em cache os embeddings das regras.

        Returns:
            Dict[str, torch.Tensor]: Dicionário de embeddings
        """
        embeddings = {}
        for key in self.rules:
            if key != "repetir":
                embeddings[key] = self.model.encode([key], convert_to_tensor=True)
        return embeddings

    def _check_casual_conversation(self, text: str) -> Optional[str]:
        """
        Verifica se o texto é uma conversa casual.

        Args:
            text (str): Texto de entrada

        Returns:
            Optional[str]: Tipo de resposta casual ou None
        """
        text = text.lower().strip()
        return next((response_type for pattern, response_type in self.casual_patterns.items()
                     if pattern in text), None)

    def _is_greeting(self, text: str) -> bool:
        """
        Verifica se o texto é uma saudação.

        Args:
            text (str): Texto de entrada

        Returns:
            bool: True se for uma saudação
        """
        greetings = {"olá", "oi", "bom dia", "boa tarde", "boa noite", "e aí",
                     "fala aí", "opa", "eae"}
        text = text.lower().strip()
        return any(text.startswith(greeting) for greeting in greetings)

    def _remove_acentos(self, texto: str) -> str:
        """Remove acentos de uma string."""
        return unidecode(texto)

    def _expandir_abreviacoes(self, texto: str) -> str:
        """Expande abreviações comuns."""
        abreviacoes = {
            "vc": "você", "pq": "porque", "oq": "o que", "eh": "é",
            "q": "que", "tb": "também", "tbm": "também", "blz": "beleza",
            "cmg": "comigo", "sao": "são", "obg": "obrigado",
            "mt": "muito", "msg": "mensagem", "agr": "agora",
            "vdd": "verdade", "mto": "muito", "to": "estou",
            "ta": "está", "n": "não", "nao": "não", "td": "tudo",
            "tá": "está", "pra": "para", "pro": "para o", "c/": "com"
            # Adicione mais abreviações conforme necessário
        }
        palavras = texto.split()
        palavras_expandidas = [abreviacoes.get(palavra, palavra) for palavra in palavras]
        return " ".join(palavras_expandidas)

    def _remover_stopwords(self, texto: str) -> str:
        """Remove stopwords de uma string."""
        palavras = [palavra for palavra in texto.split() if palavra not in self.stopwords]
        return " ".join(palavras)

    def _stemming(self, texto: str) -> str:
        """Aplica stemming às palavras de uma string."""
        palavras = [self.stemmer.stem(palavra) for palavra in texto.split()]
        return " ".join(palavras)

    def _preprocess_question(self, question: str) -> str:
        """Pré-processa a pergunta."""
        question = self._remove_acentos(question.lower())
        question = self._expandir_abreviacoes(question)
        question = self._remover_stopwords(question)
        question = self._stemming(question)
        return question

    def _find_best_response(self, question_embedding: torch.Tensor) -> Tuple[str, float]:
        """Encontra a melhor resposta com base na similaridade."""

        best_similarity = -1
        best_response = None

        for key, rule_embedding in self.rules_embeddings.items():
            similarity = util.pytorch_cos_sim(question_embedding, rule_embedding)[0].max().item()
            if similarity > best_similarity:
                best_similarity = similarity
                best_response = key

        if best_similarity < self.config.similarity_threshold:
            casual_response = self._check_casual_conversation(question_embedding)
            if casual_response:
                best_response = casual_response
                best_similarity = 1.0  # Considerar alta similaridade para conversas casuais
            else:
                best_response = "fallback"

        return random.choice(self.rules[best_response]), best_similarity


    def _is_affirmative(self, text: str) -> bool:
        """Verifica se o texto é uma resposta afirmativa."""
        affirmative_words = {"sim", "quero", "claro", "com certeza", "pode ser", "afirmativo", "ok", "beleza", "isso", "exato", "certo"}
        text = self._remove_acentos(text.lower()).strip()
        return text in affirmative_words

    def _handle_affirmative_response(self) -> Optional[str]:
        """Lida com respostas afirmativas do usuário."""
        if len(self.conversation_history) > 1:
            # Usar o tipo da última pergunta para decidir se uma elaboração é necessária
            if self.last_response_type == "pergunta_conceito":
                return self._elaborate_on_last_topic()
        return None

    def _elaborate_on_last_topic(self) -> str:
        """Fornece mais informações sobre o último tópico discutido."""
        if len(self.conversation_history) < 2:
            return "Desculpe, não tenho um tópico anterior para elaborar."

        # Obter o tipo da última pergunta registrada no histórico
        last_question_type = self.conversation_history[-2].get('type')

        if last_question_type == "pergunta_conceito":
            return self._explain_python_concepts()

        return "Desculpe, não tenho mais informações sobre esse tópico."

    def _explain_python_concepts(self) -> str:
        """Fornece uma explicação detalhada dos conceitos básicos de Python."""
        concept_details = {
            "variaveis": ("Variáveis são usadas para armazenar dados. Em Python, você não precisa declarar "
                          "explicitamente o tipo de uma variável. Exemplos: `x = 5`, `nome = 'Alice'`."),
            "listas": ("Listas são coleções ordenadas e mutáveis de itens. "
                       "Exemplo: `minha_lista = [1, 2, 'python']`."),
            "dicionarios": ("Dicionários armazenam pares de chave-valor, permitindo a recuperação rápida de valores "
                            "através de chaves. Exemplo: `d = {'nome': 'Alice', 'idade': 30}`."),
            "tuplas": ("Tuplas são como listas, mas são imutáveis, ou seja, não podem ser alteradas após a criação. "
                      "Exemplo: `minha_tupla = (1, 2, 'python')`."),
            "loops": ("Loops são usados para executar um bloco de código repetidamente. "
                     "Python tem loops `for` e `while`. Exemplo de loop `for`: `for i in range(5): print(i)`."),
            "funcoes": ("Funções são blocos de código reutilizáveis que realizam uma tarefa específica. "
                       "Exemplo: `def saudacao(nome): return 'Olá, ' + nome`."),
            "classes": ("Classes são usadas para criar objetos e são a base da programação orientada a objetos. "
                      "Exemplo: `class Cachorro: def __init__(self, nome): self.nome = nome`."),
            "condicionais": ("Condicionais permitem que o programa execute diferentes blocos de código com base em "
                             "condições. Exemplo: `if x > 0: print('Positivo') else: print('Não Positivo')`.")
        }

        # Escolhe um conceito aleatório para explicar
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
            if self._is_greeting(question):
                self.last_response_type = "saudacao"
                return random.choice(self.rules["saudacoes"])

            processed_question = self._preprocess_question(question)

            # Lidar com respostas afirmativas
            if self._is_affirmative(question) and len(self.conversation_history) > 0:
                response = self._handle_affirmative_response()
                if response:
                    return response

            question_embedding = self.model.encode(processed_question, convert_to_tensor=True)
            response, similarity = self._find_best_response(question_embedding)

            # Registrar o tipo da pergunta atual
            if "conceit" in processed_question:
                self.last_response_type = "pergunta_conceito"
            else:
                self.last_response_type = "geral"

            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'processed_question': processed_question,
                'response': response,
                'similarity_score': similarity,
                'type': self.last_response_type  # Adicionado para registrar o tipo da pergunta
            })

            return response

        except Exception as e:
            logging.error(f"Erro ao gerar resposta: {e}")
            return "Desculpe, ocorreu um erro ao processar sua pergunta. Pode tentar novamente?"

    def save_conversation(self) -> None:
        """
        Salva o histórico da conversa em arquivo JSON.
        """
        try:
            with open(self.config.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=4)
            logging.info("Histórico da conversa salvo com sucesso")
        except IOError as e:
            logging.error(f"Erro ao salvar histórico: {e}")
            print("Aviso: Não foi possível salvar o histórico da conversa")

    def run(self) -> None:
        """
        Executa o loop interativo do chatbot.
        """
        print("Olá! Eu sou um chatbot sobre Python. Pergunte-me qualquer coisa! (Digite 'sair' para encerrar)")

        try:
            while True:
                question = input("Você: ").strip()

                if question.lower() == 'sair':
                    print("Chatbot:", random.choice(self.rules["despedidas"]))
                    self.save_conversation()
                    break

                response = self.get_response(question)
                print("Chatbot:", response)

        except KeyboardInterrupt:
            print("\nEncerrando o chatbot...")
            self.save_conversation()
        except ChatbotError as e:
            logging.error(f"Erro do chatbot: {e}")
            print(f"Erro: {e}")
            self.save_conversation()

def main():
    """Função principal para executar o chatbot."""
    try:
        config = ChatbotConfig()
        chatbot = PythonChatbot(config)
        chatbot.run()
    except ChatbotError as e:
        logging.error(f"Erro fatal: {e}")
        print(f"Erro crítico: {e}")
    except Exception as e:
        logging.error(f"Erro não esperado: {e}")
        print("Ocorreu um erro inesperado. O chatbot será encerrado.")

if __name__ == "__main__":
    main()