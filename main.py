import json
import sqlite3
import re
from transformers import pipeline
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Baixar recursos do nltk
nltk.download('punkt')
nltk.download('stopwords')


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

# Função de pré-processamento do texto (normalização)
def preprocessar_texto(texto):
    """
    Pré-processa o texto removendo stopwords, aplicando tokenização e stemming.
    """
    # Tokenização
    tokens = word_tokenize(texto.lower())
    
    # Remover stopwords
    stop_words = set(stopwords.words('portuguese'))  # Certifique-se de que a lista de stopwords está no idioma correto
    tokens_filtrados = [word for word in tokens if word not in stop_words]

    # Aplicar stemming (ou lematização)
    stemmer = PorterStemmer()
    tokens_stemmed = [stemmer.stem(word) for word in tokens_filtrados]
    
    return ' '.join(tokens_stemmed)

# Função para identificar intenção usando o pipeline zero-shot
def identificar_intencao_zero_shot(mensagem_usuario, intents):
    """
    Usa um pipeline zero-shot para identificar a intenção sem treinamento customizado.
    """
    # Usar o modelo zero-shot BART pré-treinado
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Lista de intenções (tags) fornecidas do intents.json
    tags = [intent['tag'] for intent in intents['intents']]

    # Pré-processar a mensagem do usuário
    mensagem_usuario_processada = preprocessar_texto(mensagem_usuario)

    # Classificar a mensagem do usuário com base nas intenções (tags)
    result = classifier(mensagem_usuario_processada, tags)
    
    # Exibir os resultados para diagnóstico
    print("Resultados zero-shot: ", result)

    # Retornar a intenção (tag) com maior pontuação
    return result['labels'][0]

# Função aprimorada para consulta ao banco de dados com fuzzy matching
def consultar_banco_fuzzy(tag, conn):
    """
    Consulta o banco de dados com base na tag da intenção e aplica correspondência aproximada.
    """
    cursor = conn.cursor()

    # Obter todos os nomes de remédios do banco de dados
    cursor.execute("SELECT nome, quantidade FROM remedios")
    resultado = cursor.fetchall()

    if resultado:
        melhor_correspondencia = None
        maior_score = 0

        # Aplicar fuzzy matching para encontrar a melhor correspondência
        for nome, quantidade in resultado:
            score = fuzz.token_set_ratio(nome.lower(), tag.lower())
            if score > maior_score:
                maior_score = score
                melhor_correspondencia = (nome, quantidade)
        
        # Definir um limite de similaridade para considerar correspondência válida
        if melhor_correspondencia and maior_score > 75:  # Ajuste o threshold de acordo com a precisão desejada
            nome, quantidade = melhor_correspondencia
            return f"{nome} tem {quantidade} unidades em estoque (score de similaridade: {maior_score}%)."
        else:
            return "Remédio não encontrado no estoque."
    else:
        return "Nenhum remédio encontrado no banco de dados."

# Função para verificar o estoque do remédio
def verificar_estoque_remedio(mensagem_usuario, intents):
    """
    Verifica a intenção do usuário com a ajuda do pipeline zero-shot e consulta o banco de dados sobre o estoque.
    """
    try:
        # Usar o pipeline zero-shot para identificar a intenção
        tag_identificada = identificar_intencao_zero_shot(mensagem_usuario, intents)

        print(f"Tag identificada: {tag_identificada}")
        
        if tag_identificada:
            # Conectar ao banco de dados e consultar o estoque usando fuzzy matching
            conn = conectar_banco_dados()
            resposta = consultar_banco_fuzzy(tag_identificada, conn)
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
