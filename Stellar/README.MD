
## Classificação Estelar com Redes Neurais

Esta atividade consiste em criar, treinar e comparar diferentes arquiteturas de Redes Neurais Artificiais utilizando a biblioteca PyTorch para classificar objetos celestes com base em dados espectrais e físicos.

O projeto realiza a limpeza dos dados, normalização, tratamento de variáveis categóricas, e avalia 3 configurações distintas de hiperparâmetros (variando camadas, otimizadores, dropout e learning rate).

### Pré-requisitos e Dataset

Para rodar este notebook, será necessário utilizar o dataset do Kaggle:
* Dataset: SDSS - Stellar Classification Dataset - RD14 ([https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset))

que é o arquivo "star_classification.csv"

---

### Como Rodar

Siga os passos abaixo no seu terminal para clonar o repositório, instalar as dependências e abrir o ambiente.

#### 1. Clonar o repositório

```bash
git clone git@github.com:lucasarruda9/ieee_cis.git
cd Stellar
```

#### 2. Instalar as dependências
```bash
Rode o comando abaixo no terminal para instalar todas as bibliotecas de uma só vez utilizando o arquivo de requisitos já presente no repositório:
`pip install -r requirements.txt`
```

#### 3. Iniciar o Jupyter Notebook
```bash
`jupyter notebook`
```

Com Jupyter aberto, basta selecionar o arquivo "Rede_neural.ipynb" e executar as células em ordem.

Ou via python, com 
```bash
python3 Rede_neural.py
```

---

### Arquitetura e Configurações Testadas

O modelo possui uma estrutura flexível (StellarNet) baseada em camadas Lineares, ativação ReLU e Dropout. O script testa as seguintes variações para análise de desempenho:

| Configuração | Camadas Ocultas | Otimizador | Épocas | Taxa de Aprendizado (LR) | Dropout |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Config 1 | [32] | SGD | 30 | 0.01 | 0.0 |
| Config 2 | [64, 32] | Adam | 50 | 0.001 | 0.3 |
| Config 3 | [128, 64, 32] | RMSprop | 100 | 0.0005 | 0.5 |

Ao final da execução, o código exibe a acurácia de teste de cada modelo e plota gráficos comparando o comportamento do Train Loss versus o Test Loss ao longo das épocas.

O arquivo Python está atualizada e gera um .png do gráfico gerado.