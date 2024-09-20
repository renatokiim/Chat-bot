import json
import sqlite3
from transformers import pipeline

# Função para conectar ao banco de dados
def conectar_banco_dados():
    """
    Conecta ao banco de dados SQLite (ou outro banco de sua escolha).
    """
    conn = sqlite3.connect('estoque.db')  # Substitua pelo seu banco de dados
    return conn

def ler_intents(arquivo_json):
    """
    Lê o arquivo intents.json e retorna seu conteúdo como um dicionário.
    """
    with open(arquivo_json, 'r') as file:
        data = json.load(file)
    return data

# Função para identificar intenção usando o pipeline zero-shot
def identificar_intencao_zero_shot(mensagem_usuario, intents):
    """
    Usa um pipeline zero-shot para identificar a intenção sem treinamento customizado.
    """
    # Usar o modelo zero-shot BART pré-treinado
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Lista de intenções (tags) fornecidas do intents.json
    tags = [intent['tag'] for intent in intents['intents']]

    # Classificar a mensagem do usuário com base nas intenções (tags)
    result = classifier(mensagem_usuario, tags)
    
    # Exibir os resultados para diagnóstico
    print("Resultados zero-shot: ", result)

    # Retornar a intenção (tag) com maior pontuação
    return result['labels'][0]

def consultar_banco(tag, conn):
    """
    Consulta o banco de dados com base na tag da intenção.
    """
    cursor = conn.cursor()

    # Exemplo de consulta: verificar o estoque de um remédio
    cursor.execute("SELECT nome, quantidade FROM remedios WHERE nome LIKE ?", ('%' + tag + '%',))
    resultado = cursor.fetchall()

    if resultado:
        respostas = [f"{nome} tem {quantidade} unidades em estoque." for nome, quantidade in resultado]
        return "\n".join(respostas)
    else:
        return "Remédio não encontrado no estoque."

def verificar_estoque_remedio(mensagem_usuario, intents):
    """
    Verifica a intenção do usuário com a ajuda do pipeline zero-shot e consulta o banco de dados sobre o estoque.
    """
    try:
        # Usar o pipeline zero-shot para identificar a intenção
        tag_identificada = identificar_intencao_zero_shot(mensagem_usuario, intents)

        print(f"Tag identificada: {tag_identificada}")
        
        if tag_identificada:
            # Conectar ao banco de dados e consultar o estoque
            conn = conectar_banco_dados()
            resposta = consultar_banco(tag_identificada, conn)
            conn.close()
            return resposta
        else:
            return "Desculpe, não entendi o que você deseja. Pode reformular a pergunta?"
    
    except Exception as e:
        return f"Erro: {str(e)}"

def main():
    # Carregar o intents.json para obter as tags
    intents = ler_intents('intents.json')

    # Loop para consultas de estoque
    while True:
        mensagem_usuario = input("Pergunte sobre o estoque de algum remédio (ou digite 'sair' para encerrar): ")
        if mensagem_usuario.lower() == 'sair':
            print("Encerrando o programa. Até logo!")
            break
        
        resposta = verificar_estoque_remedio(mensagem_usuario, intents)
        print(resposta)

if __name__ == "__main__":
    main()
