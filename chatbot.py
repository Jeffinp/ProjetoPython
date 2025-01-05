import re
import random
from fuzzywuzzy import fuzz
from unidecode import unidecode

def remove_acentos(texto):
    """Remove acentos de uma string."""
    return unidecode(texto)

def get_resposta(pergunta, historico_conversa):
    """
    Responde a uma pergunta com base em regras, contexto e similaridade.
    """
    regras = {
        # Saudações
        r"(?i)(olá|oi|bom dia|boa tarde|boa noite)": [
            "Olá! Como posso ajudar?",
            "Oi! O que você gostaria de saber sobre Python?",
            "Olá! Bem-vindo(a) ao chatbot de Python!"
        ],
        # Despedidas
        r"(?i)(tchau|até mais|até logo|adeus)": [
            "Até mais! Espero ter ajudado.",
            "Tchau! Volte sempre que tiver dúvidas sobre Python.",
            "Foi um prazer ajudar. Até a próxima!"
        ],
        # --- Perguntas sobre Python ---
        r"(?i)(o que é|oq é|o q é|o q eh|que eh|\s*o\s*q\s*é\s*)(\s*)python\??": [
            "Python é uma linguagem de programação de alto nível, interpretada, de propósito geral, com tipagem dinâmica forte e suporte a múltiplos paradigmas, como o imperativo, orientado a objetos e funcional.",
        ],
        r"(?i)(como (começar|aprender|comecar|aprend) python|como faco pra aprender python)\??": [
            "Você pode começar aprendendo Python através de tutoriais online, cursos (como os da Udemy, Coursera, DataCamp), livros e, o mais importante, praticando bastante!",
            "Recomendo sites como Codecademy, freeCodeCamp e o próprio site oficial do Python (python.org) para iniciar sua jornada de aprendizado."
        ],
        r"(?i)(quais as (vantagens|benefícios|beneficios) de usar python|pq usar python)\??": [
            "Python é conhecido por sua sintaxe simples e legível, o que facilita o aprendizado e a escrita de código. Além disso, possui uma vasta coleção de bibliotecas para diversas aplicações, como desenvolvimento web, ciência de dados, aprendizado de máquina e automação.",
            "A grande comunidade de desenvolvedores Python é outro ponto forte, o que significa que você encontrará muita ajuda e recursos online."
        ],
        r"(?i)(o que são|o que é|oq são|oq é|o q eh|o q sao) (variáveis|variaveis|listas|dicionários|dicionarios|loops|funções|funcoes) em python\??": [
            "Esses são conceitos fundamentais em Python. Variáveis armazenam dados, listas armazenam coleções ordenadas de itens, dicionários armazenam pares chave-valor, loops permitem executar blocos de código repetidamente e funções são blocos de código reutilizáveis.",
        ],
        r"(?i)(como instalar (bibliotecas|pacotes) em python|como instala bibliotecas em python)\??": [
            "Você pode instalar bibliotecas em Python usando o gerenciador de pacotes pip. Por exemplo, para instalar a biblioteca requests, você usaria o comando 'pip install requests' no seu terminal.",
        ],
        r"(?i)(o que é|oq é|o q é|o q eh) (orientação a objetos|poo) em python\??": [
            "Orientação a Objetos é um paradigma de programação baseado no conceito de 'objetos', que podem conter dados (atributos) e código (métodos). Python suporta POO, permitindo a criação de classes e objetos.",
        ],
        r"(?i)(diferença|diferenca) entre (lista|tupla|conjunto|set) em python\??": [
            "Listas são mutáveis e ordenadas. Tuplas são imutáveis e ordenadas. Conjuntos (sets) são mutáveis e não ordenados, e não permitem elementos duplicados.",
        ],
        r"(?i)(como (ler|escrever) (arquivo|ficheiro) em python|como le e escreve arquivo em python)\??": [
            "Você pode usar a função 'open()' para ler e escrever arquivos em Python. Por exemplo: 'with open('arquivo.txt', 'r') as f: conteudo = f.read()' para ler, e 'with open('arquivo.txt', 'w') as f: f.write('texto')' para escrever.",
        ],
        # --- Perguntas que exigem contexto ---
        r"(?i)(repetir|repete)": [
            lambda c: c[-1] if c else "Desculpe, não tenho nada para repetir pois não falei nada anteriormente.",
        ],

        # --- Perguntas sobre o Chatbot ---
        r"(?i)(quem (é você|te criou|te desenvolveu|eh voce|criou voce))\??": [
            "Eu sou um chatbot criado para ajudar a responder perguntas sobre Python. Fui desenvolvido por um modelo de linguagem avançado.",
        ],
        r"(?i)(o que você pode fazer|o que vc pode fazer)\??": [
            "Eu posso responder perguntas sobre Python, te ajudar a entender conceitos da linguagem e te dar dicas de aprendizado.",
        ],
        # --- Fallback (resposta padrão) ---
        r".*": [
            "Desculpe, não tenho certeza se entendi sua pergunta. Você pode reformular ou tentar perguntar de outra forma?",
            "Ainda estou aprendendo sobre Python. Não sei responder a essa pergunta específica, mas posso tentar responder outras perguntas sobre a linguagem.",
            "Hmmm, não sei responder a essa pergunta. Você pode tentar perguntar sobre outro tópico relacionado a Python?",
        ]
    }

    # Pré-processamento da pergunta:
    pergunta = remove_acentos(pergunta.lower())  # Remove acentos e converte para minúsculas

    # Busca exata por padrão (regex)
    for padrao, respostas in regras.items():
        match = re.search(padrao, pergunta)
        if match:
            resposta = random.choice(respostas)
            if callable(resposta):
                return resposta(historico_conversa)
            else:
                return resposta
    
    # print("Nenhuma correspondência exata encontrada.")

    # Busca aproximada por similaridade (fuzzywuzzy)
    melhor_similaridade = 0
    melhor_resposta = None

    for padrao, respostas in regras.items():
        # Pré-processamento do padrão:
        padrao_sem_acentos = remove_acentos(padrao)
        # print(f"Verificando padrão: {padrao_sem_acentos}")

        similaridade = fuzz.partial_ratio(pergunta, padrao_sem_acentos)
        # print(f"Similaridade com {padrao_sem_acentos}: {similaridade}")

        if similaridade > melhor_similaridade:
            melhor_similaridade = similaridade
            melhor_resposta = random.choice(respostas)

    # Define um limite aceitável de similaridade
    if melhor_similaridade >= 75:
        # print(f"Melhor similaridade encontrada: {melhor_similaridade}")
        if callable(melhor_resposta):
            return melhor_resposta(historico_conversa)
        else:
            return melhor_resposta
    else:
        # print("Nenhuma correspondência aproximada encontrada. Retornando resposta padrão.")
        return random.choice(regras[".*"])

def main():
    """
    Função principal do chatbot.
    """
    historico_conversa = []
    print("Olá! Eu sou um chatbot sobre Python. Pergunte-me qualquer coisa!")
    while True:
        pergunta = input("Você: ")
        if pergunta.lower() == 'sair':
            break
        resposta = get_resposta(pergunta, historico_conversa)
        historico_conversa.append(resposta)

        print("Chatbot:", resposta)

if __name__ == "__main__":
    main()