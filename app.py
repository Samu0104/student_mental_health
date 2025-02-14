import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import streamlit as st

def analyze_student_mental_health(age, gender, year_of_study, marital_status):
    # Carregar o dataset
    df = pd.read_csv('student_mental_health.csv')

    # Iniciar a coleta de resultados
    results = []

    # Informações básicas
    results.append("Informações básicas do dataset:")
    results.append(df.info())

    # Dados faltantes
    missing_data = df.isnull().sum()
    results.append("\nDados faltantes:")
    results.append(missing_data)

    # Distribuição de Idade
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Age'], bins=10, kde=True, color='skyblue')
    plt.title('Distribuição de Idade dos Estudantes')
    plt.xlabel('Idade')
    plt.ylabel('Contagem')
    plt.savefig("age_distribution.png")
    plt.close()
    results.append("\nDistribuição de Idade salva em 'age_distribution.png'.")

    # Contagem de gênero
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Choose your gender', data=df, palette='Set2')
    plt.title('Distribuição por Gênero')
    plt.xlabel('Gênero')
    plt.ylabel('Contagem')
    plt.savefig("gender_distribution.png")
    plt.close()
    results.append("\nDistribuição por Gênero salva em 'gender_distribution.png'.")

    # Previsão usando KNN
    # Pré-processamento dos dados
    df['Do you have Depression?'] = df['Do you have Depression?'].map({'Yes': 1, 'No': 0})
    df['Do you have Anxiety?'] = df['Do you have Anxiety?'].map({'Yes': 1, 'No': 0})
    df['Do you have Panic attack?'] = df['Do you have Panic attack?'].map({'Yes': 1, 'No': 0})

    # Selecionar características e rótulos
    X = df[['Age', 'Choose your gender', 'Your current year of Study', 'Marital status']]
    y = df['Do you have Depression?']

    # Codificação de variáveis categóricas
    X = pd.get_dummies(X, drop_first=True)

    # Dividir os dados em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preencher valores ausentes com a média
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Treinar o modelo KNN
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Criar um DataFrame de entrada com todas as colunas esperadas
    input_data = pd.DataFrame(0, index=[0], columns=X.columns)

    # Preencher os valores fornecidos pelo usuário
    input_data['Age'] = age

    # Mapear gênero
    if gender == 'Feminino':
        if 'Choose your gender_Female' in input_data.columns:
            input_data['Choose your gender_Female'] = 1
    elif gender == 'Masculino':
        if 'Choose your gender_Male' in input_data.columns:
            input_data['Choose your gender_Male'] = 1
    elif gender == 'Outro':
        if 'Choose your gender_Other' in input_data.columns:
            input_data['Choose your gender_Other'] = 1

    # Mapear ano de estudo
    if year_of_study == 2:
        if 'Your current year of Study_Year 2' in input_data.columns:
            input_data['Your current year of Study_Year 2'] = 1
    elif year_of_study == 3:
        if 'Your current year of Study_Year 3' in input_data.columns:
            input_data['Your current year of Study_Year 3'] = 1
    elif year_of_study == 4:
        if 'Your current year of Study_Year 4' in input_data.columns:
            input_data['Your current year of Study_Year 4'] = 1
    elif year_of_study == 5:
        if 'Your current year of Study_Year 5' in input_data.columns:
            input_data['Your current year of Study_Year 5'] = 1

    # Mapear estado civil
    if marital_status == 'Casado':
        if 'Marital status_Married' in input_data.columns:
            input_data['Marital status_Married'] = 1
    elif marital_status == 'Divorciado':
        if 'Marital status_Divorced' in input_data.columns:
            input_data['Marital status_Divorced'] = 1
    elif marital_status == 'Solteiro':
        if 'Marital status_Single' in input_data.columns:
            input_data['Marital status_Single'] = 1

    # Aplicar o imputer aos dados de entrada
    input_data = imputer.transform(input_data)

    # Fazer a previsão
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        results.append("Você pode estar enfrentando sintomas de depressão.")
    else:
        results.append("Você não apresenta sintomas de depressão.")

    # Retornar resultados e gráficos
    return "\n".join(map(str, results)), "age_distribution.png", "gender_distribution.png"

# Interface do Streamlit
st.title("Análise de Saúde Mental dos Estudantes")
st.write("Análise exploratória do dataset 'Student Mental Health' e previsão de saúde mental.")

# Inputs do usuário
age = st.number_input("Idade", value=20, min_value=0)
gender = st.radio("Gênero", options=["Masculino", "Feminino", "Outro"])
year_of_study = st.selectbox("Ano de Estudo", options=[1, 2, 3, 4, 5])
marital_status = st.radio("Estado Civil", options=["Solteiro", "Casado", "Divorciado"])

if st.button("Analisar"):
    results, age_dist_img, gender_dist_img = analyze_student_mental_health(age, gender, year_of_study, marital_status)
    
    # Exibir resultados
    st.text_area("Resultados da Análise", value=results, height=300)
    
    # Exibir gráficos
    st.image(age_dist_img, caption='Distribuição de Idade')
    st.image(gender_dist_img, caption='Distribuição por Gênero')