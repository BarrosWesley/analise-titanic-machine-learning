import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def cargaDados(arquivo):
    """Carrega os dados a partir de um arquivo CSV"""
    dados = pd.read_csv(arquivo, sep=",")
    return dados

def prepararDados(dados):
    """Prepara e limpa os dados para análise"""
    print("\nInformações iniciais do dataset:")
    print(dados.info())
    print("\nPrimeiras linhas do dataset:")
    print(dados.head())
    
    # Remover dados duplicados
    dados.drop_duplicates(inplace=True)
    
    # Remover colunas com muitos nulos ou pouco informativas
    dados.drop(columns=['Cabin', 'PassengerId', 'Name', 'Ticket'], inplace=True)
    
    # Remover dados nulos
    dados.dropna(inplace=True)
    
    # Converter tipo de coluna
    dados['Age'] = dados['Age'].astype(int)
    
    # Codificar variáveis categóricas
    le = LabelEncoder()
    dados['Sex'] = le.fit_transform(dados['Sex'])
    dados['Embarked'] = le.fit_transform(dados['Embarked'])
    
    print("\nInformações após preparação:")
    print(dados.info())
    print("\nEstatísticas descritivas:")
    print(dados.describe())
    
    return dados

def dividirDados(dados, target_col, test_size=0.3, random_state=42):
    """
    Divide o dataframe em conjuntos de treino e teste
    """
    X = dados.drop(columns=[target_col])
    y = dados[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nDivisão dos dados: Treino={len(X_train)} (70%), Teste={len(X_test)} (30%)")
    
    return X_train, X_test, y_train, y_test

def treinarModelo(X_train, y_train, random_state=42):
    """
    Treina um modelo de classificação (Random Forest)
    """
    print("\nTreinando o modelo Random Forest...")
    modelo = RandomForestClassifier(random_state=random_state)
    modelo.fit(X_train, y_train)
    return modelo

def calcularAcuracia(modelo, X_test, y_test):
    """
    Calcula a acurácia do modelo
    """
    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    return acuracia

def visualizarDistribuicaoTarget(dados, target_col):
    """Visualiza a distribuição da variável target"""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_col, data=dados)
    plt.title(f'Distribuição da variável target: {target_col}')
    plt.xlabel('Sobreviveu (0 = Não, 1 = Sim)')
    plt.ylabel('Contagem')
    plt.show()

def visualizarCorrelacao(dados):
    """Matriz de correlação entre as variáveis"""
    plt.figure(figsize=(10, 8))
    corr = dados.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title('Matriz de Correlação')
    plt.show()

def visualizarDistribuicaoIdadePorSobrevivencia(dados):
    """Distribuição de idade por sobrevivência"""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=dados, x='Age', hue='Survived', bins=30, kde=True, 
                multiple='stack', palette={0: 'red', 1: 'green'})
    plt.title('Distribuição de Idade por Sobrevivência')
    plt.xlabel('Idade')
    plt.ylabel('Contagem')
    plt.legend(title='Sobreviveu', labels=['Não', 'Sim'])
    plt.show()

def visualizarClassePorSobrevivencia(dados):
    """Relação entre classe e sobrevivência"""
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Pclass', hue='Survived', data=dados, palette={0: 'red', 1: 'green'})
    plt.title('Sobrevivência por Classe')
    plt.xlabel('Classe (1 = Primeira, 2 = Segunda, 3 = Terceira)')
    plt.ylabel('Contagem')
    plt.legend(title='Sobreviveu', labels=['Não', 'Sim'])
    plt.show()

def visualizarSexoPorSobrevivencia(dados):
    """Relação entre sexo e sobrevivência"""
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Sex', hue='Survived', data=dados, palette={0: 'red', 1: 'green'})
    plt.title('Sobrevivência por Sexo')
    plt.xlabel('Sexo (0 = Feminino, 1 = Masculino)')
    plt.ylabel('Contagem')
    plt.legend(title='Sobreviveu', labels=['Não', 'Sim'])
    plt.show()

def visualizarMatrizConfusao(modelo, X_test, y_test):
    """Matriz de confusão do modelo"""
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Não Sobreviveu', 'Sobreviveu'],
                yticklabels=['Não Sobreviveu', 'Sobreviveu'])
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()

def visualizarImportanciaFeatures(modelo, features):
    """Importância das features no modelo"""
    importancia = modelo.feature_importances_
    indices = np.argsort(importancia)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Importância das Features no Modelo')
    bars = plt.bar(range(len(features)), importancia[indices], align='center', color='skyblue')
    plt.xticks(range(len(features)), np.array(features)[indices], rotation=45)
    plt.ylabel('Importância Relativa')
    plt.tight_layout()
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.show()

# Pipeline principal
if __name__ == "__main__":
    # 1. Carregar dados
    print("1. CARREGANDO OS DADOS...")
    dados = cargaDados("codigos\Titanic-Dataset.csv")
    
    # 2. Preparar os dados
    print("\n2. PREPARANDO OS DADOS...")
    dados_preparados = prepararDados(dados)
    
    # 3. Visualizações exploratórias
    print("\n3. VISUALIZAÇÕES EXPLORATÓRIAS...")
    visualizarDistribuicaoTarget(dados_preparados, 'Survived')
    visualizarCorrelacao(dados_preparados)
    visualizarDistribuicaoIdadePorSobrevivencia(dados_preparados)
    visualizarClassePorSobrevivencia(dados_preparados)
    visualizarSexoPorSobrevivencia(dados_preparados)
    
    # 4. Dividir dados em treino e teste (70%/30%)
    print("\n4. DIVIDINDO OS DADOS EM TREINO E TESTE...")
    X_train, X_test, y_train, y_test = dividirDados(dados_preparados, 'Survived')
    
    # 5. Treinar modelo
    print("\n5. TREINANDO O MODELO...")
    modelo = treinarModelo(X_train, y_train)
    
    # 6. Avaliar o modelo
    print("\n6. AVALIANDO O MODELO...")
    acuracia = calcularAcuracia(modelo, X_test, y_test)
    print(f"\nAcurácia do modelo: {acuracia:.2%}")
    
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, modelo.predict(X_test)))
    
    # 7. Visualizações do modelo
    print("\n7. VISUALIZAÇÕES DO MODELO...")
    visualizarMatrizConfusao(modelo, X_test, y_test)
    visualizarImportanciaFeatures(modelo, X_train.columns.values)
    
    print("\nANÁLISE CONCLUÍDA COM SUCESSO!")