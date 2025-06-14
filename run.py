"""
Arquivo principal para execução do chatbot Python.
"""

import os
import nltk


def check_nltk_resources():
    """Verifica se os recursos NLTK necessários estão disponíveis."""
    resources_path = os.path.join(
        os.path.expanduser("~"), "AppData", "Roaming", "nltk_data"
    )
    if not os.path.exists(resources_path):
        print("Baixando recursos NLTK necessários para o chatbot...")
        nltk.download("stopwords")
        nltk.download("rslp")
        print("Recursos NLTK baixados com sucesso!")


from src.main import main

if __name__ == "__main__":
    check_nltk_resources()
    main()
