from flask import Flask, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Carregar dados do arquivo Excel
df = pd.read_excel('mega_sena.xlsx', skiprows=7, usecols='C:H', header=None)

# Definir nomes das colunas manualmente
column_names = ['bola1', 'bola2', 'bola3', 'bola4', 'bola5', 'bola6']
df.columns = column_names

# Escolher as colunas relevantes
features = df  # Usamos todos os números sorteados como features
target = df.copy()  # A previsão é baseada nos próprios números sorteados

# Normalizar os dados
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Verificar se há um modelo previamente treinado
model_filename = 'lottery_model.h5'

if os.path.exists(model_filename):
    # Carregar o modelo previamente treinado
    model = tf.keras.models.load_model(model_filename)
else:
    # Criar o modelo de rede neural simples
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(features_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(features_scaled.shape[1])  # Saída com o mesmo número de features
    ])

    # Compilar o modelo
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Treinar o modelo
    model.fit(features_scaled, target, epochs=50, batch_size=32)

    # Salvar o modelo treinado
    model.save(model_filename)


""" def generate_lottery_suggestion():
    # Gerar sugestões para um próximo sorteio
    suggestion_scaled = model.predict(np.random.rand(1, features_scaled.shape[1]))  # Entrada aleatória para prever
    suggestion = scaler.inverse_transform(suggestion_scaled)  # Desnormalizar as sugestões
    suggestion = suggestion.flatten().astype(int).tolist()  # Converter para lista

    # Limitar os números sugeridos para o intervalo de 1 a 60
    suggestion = [np.clip(num, 1, 60) for num in suggestion]

    # Garantir que os seis números sejam distintos
    suggestion = list(set(suggestion))

    # Se o modelo gerar menos de seis números distintos, completar com números aleatórios
    while len(suggestion) < 6:
        new_number = np.random.randint(1, 61)
        if new_number not in suggestion:
            suggestion.append(new_number)

    # Se o modelo gerar mais de seis números distintos, pegar os seis primeiros
    suggestion = suggestion[:6]

    return suggestion """

"""
=========================================================
TENTATIVA DE CORREÇÃO DO NÚMERO 60 EM TODAS AS SUGESTÕES
=========================================================
"""
def generate_lottery_suggestion():
    # Obtém a contagem de ocorrências de cada número
    number_counts = df.stack().value_counts()

    # Separa os números mais sorteados e menos sorteados (25 de cada)
    most_common_numbers = number_counts.head(25).index.tolist()
    least_common_numbers = number_counts.tail(25).index.tolist()

    suggestion = []
    
    # Adiciona 3 números dos mais sorteados e 3 números dos menos sorteados
    for _ in range(3):
        suggestion.append(np.random.choice(most_common_numbers))
        suggestion.append(np.random.choice(least_common_numbers))

    return suggestion[:6]  # Retorna apenas 6 números


"""
===============================================================
FIM DA TENTATIVA DE CORREÇÃO DO NÚMERO 60 EM TODAS AS SUGESTÕES
===============================================================
"""

def plot_lottery_frequency():
    # Criar uma lista com todos os números sorteados
    all_numbers = df.values.flatten()

    # Criar o gráfico de frequência
    plt.figure(figsize=(10, 6))
    plt.hist(all_numbers, bins=range(1, 62), align='left', edgecolor='black', alpha=0.7)
    plt.title('Frequência dos Números Sorteados na Mega Sena')
    plt.xlabel('Número Sorteado')
    plt.ylabel('Frequência')
    plt.xticks(range(1, 61))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Salvar a imagem em BytesIO
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.read()).decode('utf-8')
    plt.close()
    # Retorna a imagem em base64 para ser exibida no template
    return img_base64


@app.route('/')
def index():
    return render_template('lottery_suggestion.html', suggestion=None)


@app.route('/lottery_suggestion', methods=['POST'])
def lottery_suggestion():
    suggestion = generate_lottery_suggestion()
    img_base64 = plot_lottery_frequency()

    return render_template('lottery_suggestion.html', suggestion=suggestion, img_base64=img_base64)


if __name__ == '__main__':
    app.run(debug=True)
