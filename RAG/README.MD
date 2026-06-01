## Pipeline RAG

Esta atividade consiste em criar, rodar e comparar o impacto de diferentes tamanhos de blocos com chunk em um pipeline de RAG, utilizando FAISS para busca vetorial e o modelo Flan-T5 para geração de texto.

O projeto realiza a captura de dados via streaming, tokenização, divisão em chunks de 256 e 512 tokens com overlap, indexação semântica e avalia 3 cenários distintos de resposta (Closed-Book, RAG 256 e RAG 512) para medir a precisão do modelo.

### Pré-requisitos e Dataset

Para rodar este script, o pipeline utiliza o seguinte dataset diretamente via Hugging Face (em modo streaming, sem necessidade de baixar arquivos pesados):
* Dataset: Wikimedia Wikipedia PT (Amostra de 73 artigos da partição `20231101.pt`)

---

### Como Rodar

Siga os passos abaixo no seu terminal para clonar o repositório, instalar as dependências e executar o experimento.

#### 1. Clonar o repositório
```bash
git clone git@github.com:lucasarruda9/ieee_cis.git
cd RAG
```
#### 2. Instalar dependências

Primeiro crie um ambiente virtual por meio de:

```bash
python3 -m venv venv
source venv/bin/activate
```

Depois isntale as dependências com:

```bash
`pip install -r requisitos.txt`
```

#### 3. Rodar projeto

Rode o projeto com o comando:

```bash
python3 pipelineRAG.py
```
### Saída do Terminal

Depois que o script terminar de rodar, irá aparecer três respostas bem diferentes na tela.

#### 1. Resposta Sem Recuperação (Closed-Book)
O modelo tenta responder sem ler texto nenhum. Como ele não sabe a resposta, ele apenas repete a própria pergunta de volta na tela.

#### 2. RAG com Chunks de 256 tokens
O modelo erra a resposta e começa a falar sobre "crítica de arte".

#### 3. RAG com Chunks de 512 tokens
O modelo acerta a resposta e explica o ponto de vista de Baudelaire sobre a era industrial.