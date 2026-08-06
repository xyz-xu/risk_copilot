import numpy as np
from scipy.sparse import csr_matrix
from rag_core.rag_factory import get_milvus_client
from rag_core.graph_rag import text_persist
from rag_core.graph_rag import text_chunk_proc

def get_entitys_len():
    client = get_milvus_client()
    res = client.query(collection_name=text_chunk_proc.entity_col_name, output_fields=["count(*)"])
    return res[0]['count(*)']

def get_relations_len():
    client = get_milvus_client()
    res = client.query(collection_name=text_chunk_proc.relation_col_name, output_fields=["count(*)"])
    return res[0]['count(*)']

def load_adj():
    entitys_len = get_entitys_len()
    relations_len = get_relations_len()
    entityid_2_relationids = text_persist.load(text_persist.type_entityid_2_relationids)

    entity_relation_adj = np.zeros((entitys_len, relations_len))
    for entity_id in range(entitys_len):
        entity_relation_adj[entity_id, entityid_2_relationids[entity_id]] = 1

    entity_relation_adj = csr_matrix(entity_relation_adj)

    entity_adj_1_degree = entity_relation_adj @ entity_relation_adj.T
    relation_adj_1_degree = entity_relation_adj.T @ entity_relation_adj

    target_degree = 1

    entity_adj_target_degree = entity_adj_1_degree
    for _ in range(target_degree - 1):
        entity_adj_target_degree = entity_adj_target_degree @ entity_adj_1_degree
    relation_adj_target_degree = relation_adj_1_degree
    for _ in range(target_degree - 1):
        relation_adj_target_degree = relation_adj_target_degree @ relation_adj_1_degree

    entity_relation_adj_target_degree = entity_adj_target_degree @ entity_relation_adj

    return relation_adj_target_degree, entity_relation_adj_target_degree

def expand_relations_by_hit(hit_relation_ids, hit_entity_ids):
    expanded_relations_from_relation = set()
    expanded_relations_from_entity = set()
    relation_adj_target_degree, entity_relation_adj_target_degree = load_adj()

    for hit_relation_id in hit_relation_ids:
        expanded_relations_from_relation.update(
            relation_adj_target_degree[hit_relation_id].nonzero()[1].tolist()
        )

    for filtered_hit_entity_id in hit_entity_ids:
        expanded_relations_from_entity.update(
            entity_relation_adj_target_degree[filtered_hit_entity_id].nonzero()[1].tolist()
        )

    relation_candidate_ids = list(
        expanded_relations_from_relation | expanded_relations_from_entity
    )

    client = get_milvus_client()

    # [{id:0,text:"xxx"}]
    res = client.query(collection_name=text_chunk_proc.relation_col_name, ids=relation_candidate_ids, output_fields=["text"])
    return res

def get_passages_by_relatioin_ids(relation_ids):
    relationid_2_passageids = text_persist.load(text_persist.type_relationid_2_passageids)

    final_passage_ids = []
    for relation_id in relation_ids:
        for passage_id in relationid_2_passageids[relation_id]:
            if passage_id not in final_passage_ids:
                final_passage_ids.append(passage_id)

    client = get_milvus_client()
    # [{id:0,text:"xxx"}]
    res = client.query(collection_name=text_chunk_proc.passage_col_name, ids=final_passage_ids, output_fields=["text"])
    return res        
