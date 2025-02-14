# data.py
import pandas as pd

def load_and_preprocess_data():
    # Carregando o dataset de saúde mental
    df = pd.read_csv("student_mental_health.csv")

    # Pré-processamento dos dados
    df['Age'].fillna(df['Age'].median(), inplace=True)  # Preenchendo valores faltantes na idade

    # Convertendo variáveis categóricas em variáveis dummy
    df = pd.get_dummies(df, columns=[
        'Choose your gender', 
        'What is your course?', 
        'Your current year of Study', 
        'What is your CGPA?', 
        'Marital status'
    ], drop_first=True)

    # Selecionando features e target
    X = df.drop(columns=[
        'Do you have Depression?', 
        'Do you have Anxiety?', 
        'Do you have Panic attack?', 
        'Did you seek any specialist for a treatment?', 
        'MEDV', 
        'Timestamp'
    ])
    y = df['Do you have Depression?']

    return X, y