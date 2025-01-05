import os
import shutil

def organizar_arquivos(diretorio):
    """
    Organiza os arquivos em um diretório, agrupando-os por tipo em subpastas.
    """
    tipos = {
        "Imagens": [".jpg", ".jpeg", ".png", ".gif", ".bmp"],
        "Documentos": [".pdf", ".doc", ".docx", ".txt", ".odt", ".xlsx", ".csv"],
        "Músicas": [".mp3", ".wav", ".ogg", ".flac"],
        "Vídeos": [".mp4", ".avi", ".mkv", ".mov"],
        # Adicione mais tipos conforme necessário
    }

    for nome_arquivo in os.listdir(diretorio):
        caminho_arquivo = os.path.join(diretorio, nome_arquivo)

        # Ignora diretórios, processa apenas arquivos
        if os.path.isfile(caminho_arquivo):
            extensao = os.path.splitext(nome_arquivo)[1].lower()

            encontrou_tipo = False
            for tipo, extensoes in tipos.items():
                if extensao in extensoes:
                    pasta_destino = os.path.join(diretorio, tipo)
                    # Cria a pasta de destino se não existir
                    os.makedirs(pasta_destino, exist_ok=True)
                    # Move o arquivo para a pasta de destino
                    shutil.move(caminho_arquivo, pasta_destino)
                    print(f"Movido '{nome_arquivo}' para '{tipo}'")
                    encontrou_tipo = True
                    break

            if not encontrou_tipo:
                pasta_destino = os.path.join(diretorio, "Outros")
                os.makedirs(pasta_destino, exist_ok=True)
                shutil.move(caminho_arquivo, pasta_destino)
                print(f"Movido '{nome_arquivo}' para 'Outros'")

if __name__ == "__main__":
    diretorio_alvo = input("Digite o caminho do diretório que deseja organizar: ")
    # Verifica se o diretório informado é um caminho válido
    if os.path.isdir(diretorio_alvo):
        organizar_arquivos(diretorio_alvo)
    else:
        print("O caminho especificado não é um diretório válido.")