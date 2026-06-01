import numpy as np
import faiss
import torch
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

print("Carregando amostra do dataset da Wikipedia...")
dataset_stream = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train", streaming=True)
dataset = list(dataset_stream.take(73))

TAMANHOS_CHUNK = [256, 512]
OVERLAP = 50

tokenizer_llm = AutoTokenizer.from_pretrained("google/flan-t5-base")

def criar_chunks(dados, tamanho_chunk, overlap):
    documentos_processados = []
    for artigo in dados:
        texto = artigo['text']
        tokens = tokenizer_llm.encode(texto)
        i = 0
        while i < len(tokens):
            fim = min(i + tamanho_chunk, len(tokens))
            chunk_tokens = tokens[i:fim]
            chunk_texto = tokenizer_llm.decode(chunk_tokens, skip_special_tokens=True)
            
            documentos_processados.append({
                "texto": chunk_texto,
                "titulo": artigo['title']
            })
            if fim == len(tokens):
                break
            i += (tamanho_chunk - overlap)
    return documentos_processados

print("Carregando modelos para vetorização e geração...")
modelo_embed = SentenceTransformer('all-MiniLM-L6-v2')
model_t5 = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def gerar_texto(prompt):
    inputs = tokenizer_llm(prompt, return_tensors="pt")
    outputs = model_t5.generate(**inputs, max_new_tokens=100)
    return tokenizer_llm.decode(outputs[0], skip_special_tokens=True)

def construir_indexador(documentos):
    textos = [doc["texto"] for doc in documentos]
    vetores = modelo_embed.encode(textos, show_progress_bar=False)
    
    dimensao = vetores.shape[1]
    indice = faiss.IndexFlatL2(dimensao)
    indice.add(np.array(vetores).astype('float32'))
    return indice

def executar_pipeline_rag(pergunta, documentos, indice, k_vizinhos=2):
    vetor_pergunta = modelo_embed.encode([pergunta]).astype('float32')
    _, indices_proximos = indice.search(vetor_pergunta, k_vizinhos)
    
    contexto_recuperado = ""
    for idx in indices_proximos[0]:
        if idx < len(documentos):
            contexto_recuperado += f" {documentos[idx]['texto']}\n"
    
    prompt = f"Responda a pergunta com base no texto fornecido.\n\nTexto: {contexto_recuperado}\n\nPergunta: {pergunta}"
    
    resposta = gerar_texto(prompt)
    return resposta, contexto_recuperado

def executar_closed_book(pergunta):
    prompt = f"Responda de forma direta: {pergunta}"
    resposta = gerar_texto(prompt)
    return resposta

pergunta_teste = "Qual a relação entre arte e o progresso industrial de acordo com Charles Baudelaire?"

print("\n--- Processando Estruturas de Chunks ---")
docs_256 = criar_chunks(dataset, 256, OVERLAP)
indice_256 = construir_indexador(docs_256)

docs_512 = criar_chunks(dataset, 512, OVERLAP)
indice_512 = construir_indexador(docs_512)

print("\n=== RESULTADOS DO EXPERIMENTO ===")
print(f"Pergunta feita: {pergunta_teste}\n")

res_closed = executar_closed_book(pergunta_teste)
print(f"1. Resposta Sem Recuperação (Closed-Book):\n> {res_closed}\n")

res_rag_256, _ = executar_pipeline_rag(pergunta_teste, docs_256, indice_256)
print(f"2. RAG com Chunks de 256 tokens:\n> {res_rag_256}\n")

res_rag_512, _ = executar_pipeline_rag(pergunta_teste, docs_512, indice_512)
print(f"3. RAG com Chunks de 512 tokens:\n> {res_rag_512}\n")