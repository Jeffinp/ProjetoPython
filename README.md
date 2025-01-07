# Python Chatbot - Assistente de Aprendizado

[![Imagem do Chatbot](https://github.com/Jeffinp/jefersonreis.github.io/blob/main/src/image/Chatbot.webp?raw=true)](https://github.com/Jeffinp/ProjetoPython/tree/main)

## Descrição

Este projeto é um chatbot desenvolvido em Python que serve como uma ferramenta de auxílio no aprendizado e consulta rápida sobre a linguagem de programação Python. Ele foi criado para ser um recurso útil tanto para iniciantes quanto para programadores experientes que desejam aprimorar seus conhecimentos ou esclarecer dúvidas.

O chatbot utiliza técnicas de Processamento de Linguagem Natural (PLN) para entender as perguntas dos usuários e fornecer respostas relevantes e detalhadas. A base do seu funcionamento é o modelo de linguagem `rufimelo/bert-large-portuguese-cased-sts`, da biblioteca `Sentence Transformers`, treinado especificamente para similaridade semântica na língua portuguesa. Isso permite que o chatbot interprete uma ampla gama de perguntas formuladas de diferentes maneiras.

Além disso, o projeto emprega as bibliotecas `NLTK` (Natural Language Toolkit) para tarefas de pré-processamento de texto, como *stemming* (redução de palavras ao seu radical) e remoção de *stopwords* (palavras comuns como "de", "a", "o", etc.), e `unidecode` para normalizar a acentuação, garantindo uma melhor compreensão das consultas.

## Funcionalidades

*   **Respostas a perguntas sobre Python:** O chatbot responde a perguntas sobre sintaxe, conceitos, bibliotecas e melhores práticas da linguagem.
*   **Explicações detalhadas:** Fornece explicações claras e exemplos práticos para facilitar o entendimento.
*   **Suporte a diferentes níveis de conhecimento:** Aborda tópicos desde o básico até o avançado, atendendo a um público amplo.
*   **Dicas de instalação e configuração:** Orienta os usuários sobre como instalar o Python, configurar ambientes virtuais e instalar pacotes.
*   **Sugestões de recursos de aprendizado:** Indica sites, livros, tutoriais e comunidades online para aprofundar o conhecimento.
*   **Interface interativa:** Permite uma conversa natural e intuitiva com o usuário.

## Tecnologias Utilizadas

*   **Python:** Linguagem de programação principal utilizada no desenvolvimento.
*   **Sentence Transformers:** Biblioteca para geração de embeddings de sentenças.
    *   **Modelo:** `rufimelo/bert-large-portuguese-cased-sts` - Modelo de linguagem para similaridade semântica em português.
*   **NLTK (Natural Language Toolkit):** Biblioteca para processamento de linguagem natural.
*   **unidecode:** Biblioteca para transliteração de texto (remoção de acentos).
*   **PyTorch:** Framework de aprendizado de máquina usado pela `Sentence Transformers`.

## Como Executar o Projeto

**Pré-requisitos:**

*   Python 3.7+
*   pip (gerenciador de pacotes do Python)

**Passos:**

1.  **Clone o repositório:**

    ```bash
    git clone [URL inválido removido]
    cd ProjetoPython
    ```

2.  **Crie um ambiente virtual (recomendado):**

    ```bash
    python3 -m venv .venv
    ```

3.  **Ative o ambiente virtual:**

    *   **Linux/macOS:**
        ```bash
        source .venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```

4.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Baixe os recursos do NLTK (se necessário):**
    *   Execute o script Python uma vez, ele vai baixar automaticamente.

6.  **Execute o chatbot:**

    ```bash
    python chatbot.py
    ```

7.  **Siga as instruções no console para interagir com o chatbot.**

## Arquivo `rules.json`

O arquivo `rules.json` contém as regras que definem as respostas do chatbot para diferentes perguntas. Ele é organizado em categorias e subcategorias, facilitando a expansão e manutenção.

**Exemplo de estrutura do `rules.json`:**

```json
{
  "saudacoes": [
    "Olá!",
    "Oi!"
  ],
  "despedidas": [
    "Tchau!",
    "Até mais!"
  ],
  "sobre_python": {
    "o_que_e_python": [
      "Python é uma linguagem de programação de alto nível..."
    ]
  },
  "conceitos_basicos": {
    "o_que_sao_variaveis": [
      "Variáveis são usadas para armazenar dados na memória..."
    ]
  }
}
