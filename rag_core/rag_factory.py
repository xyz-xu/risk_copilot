# milvus client
from pymilvus import MilvusClient
from rag_core.rag_constants import *

_milvus_client = MilvusClient(
    uri=milvus_uri,
    db_name=milvus_db_name,
    user=milvus_user,
    password=milvus_password
)

def get_milvus_client():
    return _milvus_client;

# embedding
from langchain_ollama import OllamaEmbeddings

_embedding_model = OllamaEmbeddings(model="qwen3-embedding")

def get_embedding_model():
    return _embedding_model

# reranker
import torch
from modelscope import AutoModelForSequenceClassification, AutoTokenizer

_reanker_tokenizer = AutoTokenizer.from_pretrained('D:\\code_files\\modelscope\\BAAI\\bge-reranker-v2-m3')
_reranker_model = AutoModelForSequenceClassification.from_pretrained('D:\\code_files\\modelscope\\BAAI\\bge-reranker-v2-m3')
_reranker_model.eval()

def get_reanker_tokenizer():
    return _reanker_tokenizer

def get_reranker_model():
    return _reranker_model;
