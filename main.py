import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


categorical_features = ['experience_level', 'employment_type', 'salary_currency', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']

data = pd.read_csv('ds_salaries.csv')

data['classificacao'] = 0
X = data[['salary_in_usd'] + categorical_features]
y_classificacao = data['classificacao']  # substitua 'classificacao' pelo nome da coluna que contém as classes alvo
y_salario = data['salary']

label_encoder = LabelEncoder()
for col in categorical_features:
    X[col] = label_encoder.fit_transform(X[col])

# Padronizar os recursos numéricos
scaler = StandardScaler()
X[['salary_in_usd']] = scaler.fit_transform(X[['salary_in_usd']])
print(X.head())
X_train, X_test, y_train_classificacao, y_test_classificacao = train_test_split(X, y_classificacao, test_size=0.2, random_state=42)

mlp_classificacao = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500)
mlp_classificacao.fit(X_train, y_train_classificacao)

# Suponha que você tenha um novo profissional representado como um dicionário
new_profissional = {
    'experience_level': 'MI',
    'employment_type': 'CT',
    'salary_currency': 'USD',
    'salary_in_usd': 80000,
    'employee_residence': 'US',
    'remote_ratio': '0',
    'company_location': 'US',
    'company_size': 'L'
}

# Pré-processe o novo profissional
for col in categorical_features:
    print(new_profissional)
    new_profissional[col] = label_encoder.transform([new_profissional.get(col)])[0]

new_profissional[['salary_in_usd']] = scaler.transform([new_profissional[['salary_in_usd']]])


# Faça a previsão de classificação
classification_prediction = mlp_classificacao.predict([list(new_profissional.values())])[0]
print('Classificação:', classification_prediction)

X_train, X_test, y_train_salario, y_test_salario = train_test_split(X, y_salario, test_size=0.2, random_state=42)

mlp_salario = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500)
mlp_salario.fit(X_train, y_train_salario)

salary_prediction = mlp_salario.predict([list(new_profissional.values())])[0]
print('Salário sugerido:', salary_prediction)

