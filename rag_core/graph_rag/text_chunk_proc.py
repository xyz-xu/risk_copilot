from collections import defaultdict

from tqdm import tqdm

from rag_core.rag_factory import get_milvus_client, get_embedding_model
from rag_core.rag_constants import embedding_dim
import rag_core.graph_rag.text_persist as text_persist

def load_text_lines():
    # 调用text_parser将文本拆分
    return [
        "Jakob Bernoulli (1654–1705): Jakob was one of the earliest members of the Bernoulli family to gain prominence in mathematics. He made significant contributions to calculus, particularly in the development of the theory of probability. He is known for the Bernoulli numbers and the Bernoulli theorem, a precursor to the law of large numbers. He was the older brother of Johann Bernoulli, another influential mathematician, and the two had a complex relationship that involved both collaboration and rivalry.",
        "Johann Bernoulli (1667–1748): Johann, Jakob’s younger brother, was also a major figure in the development of calculus. He worked on infinitesimal calculus and was instrumental in spreading the ideas of Leibniz across Europe. Johann also contributed to the calculus of variations and was known for his work on the brachistochrone problem, which is the curve of fastest descent between two points.",
        "Daniel Bernoulli (1700–1782): The son of Johann Bernoulli, Daniel made major contributions to fluid dynamics, probability, and statistics. He is most famous for Bernoulli’s principle, which describes the behavior of fluid flow and is fundamental to the understanding of aerodynamics.",
    ]

def load_triplets(lists: list):
    triplets=[]
    for index, doc in enumerate(lists):
        # 调用大模型生成三元组
        triplet = []
        if index == 0:
            triplet = [
                ["Jakob Bernoulli", "made significant contributions to", "calculus"],
                [
                    "Jakob Bernoulli",
                    "made significant contributions to",
                    "the theory of probability",
                ],
                ["Jakob Bernoulli", "is known for", "the Bernoulli numbers"],
                ["Jakob Bernoulli", "is known for", "the Bernoulli theorem"],
                ["The Bernoulli theorem", "is a precursor to", "the law of large numbers"],
                ["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"],
            ]
        elif index == 1:
            triplet = [
                [
                    "Johann Bernoulli",
                    "was a major figure of",
                    "the development of calculus",
                ],
                ["Johann Bernoulli", "was", "Jakob's younger brother"],
                ["Johann Bernoulli", "worked on", "infinitesimal calculus"],
                ["Johann Bernoulli", "was instrumental in spreading", "Leibniz's ideas"],
                ["Johann Bernoulli", "contributed to", "the calculus of variations"],
                ["Johann Bernoulli", "was known for", "the brachistochrone problem"],
            ]
        elif index == 2:
            triplet = [
                ["Daniel Bernoulli", "was the son of", "Johann Bernoulli"],
                ["Daniel Bernoulli", "made major contributions to", "fluid dynamics"],
                ["Daniel Bernoulli", "made major contributions to", "probability"],
                ["Daniel Bernoulli", "made major contributions to", "statistics"],
                ["Daniel Bernoulli", "is most famous for", "Bernoulli’s principle"],
                [
                    "Bernoulli’s principle",
                    "is fundamental to",
                    "the understanding of aerodynamics",
                ],
            ]
        elif index == 3:
            triplet = [
                [
                    "Leonhard Euler",
                    "had a significant relationship with",
                    "the Bernoulli family",
                ],
                ["leonhard Euler", "was born in", "Basel"],
                ["Leonhard Euler", "was a student of", "Johann Bernoulli"],
                ["Johann Bernoulli's influence", "was profound on", "Euler"],
            ]
        triplets.append(triplet)
    return triplets

def load_dataset():
    """载入数据，并整合成dataset"""
    lists = load_text_lines()
    triplets = load_triplets(lists)
    dataset = [{"passage": passage, "triplets": triplets} for (passage, triplets) in zip(lists, triplets)]
    return dataset

def calc_relations(dataset):
    """计算关系"""
    entityid_2_relationids = defaultdict(list)
    relationid_2_passageids = defaultdict(list)

    entities = []
    relations = []
    passages = []
    for passage_id, dataset_info in enumerate(dataset):
        passage, triplets = dataset_info["passage"], dataset_info["triplets"]
        passages.append(passage)
        for triplet in triplets:
            if triplet[0] not in entities:
                entities.append(triplet[0])
            if triplet[2] not in entities:
                entities.append(triplet[2])
            relation = " ".join(triplet)
            if relation not in relations:
                relations.append(relation)
                entityid_2_relationids[entities.index(triplet[0])].append(
                    len(relations) - 1
                )
                entityid_2_relationids[entities.index(triplet[2])].append(
                    len(relations) - 1
                )
            relationid_2_passageids[relations.index(relation)].append(passage_id)
    return entityid_2_relationids, relationid_2_passageids, entities, relations, passages

def create_milvus_collection(client, collection_name):
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        consistency_level="Bounded",
    )

entity_col_name = "entity_collection"
relation_col_name = "relation_collection"
passage_col_name = "passage_collection"

def init_collections(client):
    create_milvus_collection(client, entity_col_name)
    create_milvus_collection(client, relation_col_name)
    create_milvus_collection(client, passage_col_name)

def milvus_insert(
        client,
        embedding_model,
        collection_name,
        text_list,
):
    batch_size = 512
    for row_id in tqdm(range(0, len(text_list), batch_size), desc="Inserting"):
        batch_texts = text_list[row_id : row_id + batch_size]
        batch_embeddings = embedding_model.embed_documents(batch_texts)

        batch_ids = [row_id + j for j in range(len(batch_texts))]
        batch_data = [
            {
                "id": id_,
                "text": text,
                "vector": vector,
            }
            for id_, text, vector in zip(batch_ids, batch_texts, batch_embeddings)
        ]
        client.insert(
            collection_name=collection_name,
            data=batch_data,
        )

def persist_to_db(entityid_2_relationids, relationid_2_passageids, entities, relations, passages):
    # vector
    client = get_milvus_client()
    embedding_model = get_embedding_model()
    init_collections(client)
    milvus_insert(client, embedding_model, relation_col_name, text_list=relations)
    milvus_insert(client, embedding_model, entity_col_name, text_list=entities)
    milvus_insert(client, embedding_model, passage_col_name, text_list=passages)

    # relations
    text_persist.save(text_persist.type_entityid_2_relationids, entityid_2_relationids)
    text_persist.save(text_persist.type_relationid_2_passageids, relationid_2_passageids)

def run():
    dataset = load_dataset()
    entityid_2_relationids, relationid_2_passageids, entities, relations, passages = calc_relations(dataset)
    persist_to_db(entityid_2_relationids, relationid_2_passageids, entities, relations, passages)
