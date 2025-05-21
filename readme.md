# Análise do Dataset Titanic - Predição de Sobrevivência

![Titanic](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1200px-RMS_Titanic_3.jpg)

Projeto de análise de dados e machine learning utilizando o dataset Titanic para prever a sobrevivência de passageiros.

## 📌 Visão Geral

Este projeto realiza uma análise completa do famoso dataset Titanic, incluindo:
- Limpeza e preparação dos dados
- Análise exploratória com visualizações
- Treinamento de modelo de machine learning
- Avaliação de desempenho do modelo

## 📊 Dados Utilizados

Dataset: [Titanic-Dataset.csv](https://www.kaggle.com/c/titanic/data)

Variáveis incluídas:
- **Survived**: Sobreviveu (0 = Não, 1 = Sim) - **target**
- **Pclass**: Classe do ticket (1 = 1ª classe, 2 = 2ª classe, 3 = 3ª classe)
- **Sex**: Sexo do passageiro
- **Age**: Idade
- **SibSp**: Número de irmãos/cônjuges a bordo
- **Parch**: Número de pais/filhos a bordo
- **Fare**: Tarifa paga
- **Embarked**: Porto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)

## 🛠️ Tecnologias Utilizadas

- Python 3.x
- Bibliotecas:
  - Pandas
  - NumPy
  - Matplotlib
  - Seaborn
  - Scikit-learn

## 🚀 Como Executar o Projeto

1. Clone o repositório:

```bash
    git clone https://github.com/
    cd titanic-analysis
    ```

## Instale as dependências:


```bash
        pip install -r requirements.txt
        ```

## 📋 Estrutura do Código
    - script principal (titanicia.py) contém as seguintes funções:
    - Preparação dos Dados:
    - cargaDados() 
    - Carrega o dataset
    - prepararDados() 
    - Limpeza e transformação dos dados
    - Análise Exploratória:
    - Visualizações da distribuição dos dados
    - Matriz de correlação
    - Relações entre variáveis
    - Modelagem Preditiva:
        - dividirDados() 
        - Divisão treino/teste (70%/30%)
        - treinarModelo() 
        - Random Forest Classifier
        - calcularAcuracia() 
        - Avaliação do modelo

## Visualizações:

    - Gráficos de distribuição
    - Matriz de confusão
    - Importância das features

## 📈 Resultados

O modelo Random Forest alcançou uma acurácia de aproximadamente 80% (pode variar dependendo da execução).

Principais insights:
    - Mulheres tiveram maior taxa de sobrevivência
    - Passageiros da 1ª classe tiveram mais chances de sobreviver
    - Crianças tiveram prioridade nos botes salva-vidas

## 🤝 Como Contribuir
    - Faça um fork do projeto
    - Crie uma branch para sua feature (git checkout -b feature/AmazingFeature)
    - Commit suas mudanças (git commit -m 'Add some AmazingFeature')
    - Push para a branch (git push origin feature/AmazingFeature)
    - Abra um Pull Request

## 📄 Licença
Distribuído sob a licença MIT. 