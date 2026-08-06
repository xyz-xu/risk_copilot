from rag_core.rag_factory import get_embedding_model, get_milvus_client, get_reranker_model, get_reanker_tokenizer
from typing import Tuple,List
from rag_core.rag_constants import *
import torch
from pymilvus import AnnSearchRequest, WeightedRanker

def rag_query(
        query: str,
        collection_name: str,
        partition_name: str = None,
        user_role: str = None,
    ) -> List[Tuple[str, str]]:
    """
    通过知识库查询相关的知识数据（向量检索）
    query - 查询问题
    collection_name - collection名称
    partition_name - partition名称
    user_role - 角色名称
    return - [(source, text) ...]
    """
    # embedding_model
    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(query)

    # client
    client = get_milvus_client()
    res = client.search(
        collection_name=collection_name,
        partition_names=[partition_name] if partition_name else None,
        filter=f'array_contains(granted_roles,"{user_role}")' if user_role else None,
        anns_field=milvus_anns_field,
        data=[query_vector],
        limit=milvus_limit,
        search_params={"metric_type": "IP", "radius": milvus_filtered_radius},
        output_fields=["text", "source_uri"]
    )

    if len(res[0]) <= 0:
        return []

    results = res[0]
    pairs = [[query, doc["entity"]["text"]] for doc in results]

    # ranker
    tokenizer = get_reanker_tokenizer()
    reranker_model = get_reranker_model()
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
    
    ranked_docs = sorted(zip(scores, results), reverse=True)
    ranked_docs = ranked_docs[:reranker_limit]
    
    llm_friendly_docs = _sort_docs_for_llm(ranked_docs)
    return llm_friendly_docs


def _sort_docs_for_llm(lists):
    """将0,1,2,3,4,5,6重新排序为0,2,4,6,5,3,1返回，因为LLM对两边的识别效果更好"""
    evens = lists[::2]
    odds = lists[1::2]
    return evens + odds[::-1]

def sparse_search(
        query: str,
        collection_name: str,
        partition_name: str = None,
        user_role: str = None,
    ) -> List[Tuple[str, str]]:
    """
    通过知识库查询相关的知识数据（全文检索）
    query - 查询问题
    collection_name - collection名称
    partition_name - partition名称
    user_role - 角色名称
    return - [(source, text) ...]
    """

    # client
    client = get_milvus_client()

    res = client.search(
        collection_name=collection_name, 
        partition_names=[partition_name] if partition_name else None,
        filter=f'array_contains(granted_roles,"{user_role}")' if user_role else None,
        anns_field=milvus_sparse_anns_field,
        data=[query],
        output_fields=["text", "source_uri"],
        limit=reranker_limit,
    )

    if len(res[0]) <= 0:
        return []
    
    results = res[0]
    scores = [r.score for r in results]

    ranked_docs = sorted(zip(scores, results), reverse=True)
        
    llm_friendly_docs = _sort_docs_for_llm(ranked_docs)
    return llm_friendly_docs

def hybrid_search(
        query: str,
        collection_name: str,
        partition_name: str = None,
        user_role: str = None,
    ) -> List[Tuple[str, str]]:
    """
    通过知识库查询相关的知识数据（混合检索）
    query - 查询问题
    collection_name - collection名称
    partition_name - partition名称
    user_role - 角色名称
    return - [(source, text) ...]
    """
    
    # embedding_model
    embedding_model = get_embedding_model()
    query_vector = embedding_model.embed_query(query)

    # dense_req
    dense_req = AnnSearchRequest(
        filter=f'array_contains(granted_roles,"{user_role}")' if user_role else None,
        anns_field=milvus_anns_field,
        data=[query_vector],
        limit=milvus_limit,
        param={"metric_type": "IP", "radius": milvus_filtered_radius},
    )

    # sparse_req
    sparse_req = AnnSearchRequest(
        filter=f'array_contains(granted_roles,"{user_role}")' if user_role else None,
        anns_field=milvus_sparse_anns_field,
        data=[query],
        limit=milvus_limit,
        param={}
    )

    # ranker
    weighted_ranker = WeightedRanker(milvus_dense_weight, milvus_sparse_weight)

    
    # client
    client = get_milvus_client()

    # hybrid_search
    res = client.hybrid_search(
        collection_name=collection_name,
        partition_names=[partition_name] if partition_name else None,
        reqs=[dense_req, sparse_req],
        ranker=weighted_ranker,
        output_fields=["text", "source_uri"],
        limit=milvus_limit,
    )

    if len(res[0]) <= 0:
        return []

    results = res[0]
    pairs = [[query, doc["entity"]["text"]] for doc in results]

    # ranker
    tokenizer = get_reanker_tokenizer()
    reranker_model = get_reranker_model()
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()
    
    ranked_docs = sorted(zip(scores, results), reverse=True)
    ranked_docs = ranked_docs[:reranker_limit]
    
    llm_friendly_docs = _sort_docs_for_llm(ranked_docs)
    return llm_friendly_docs
