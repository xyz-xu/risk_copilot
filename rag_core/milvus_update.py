from rag_core.rag_factory import get_milvus_client

def update():
    client = get_milvus_client()
    results = client.query(
        collection_name="risk_knowledge_base",
        filter='array_contains(granted_roles,"user_001")',
        limit=100
    )

    for hit in results:
        hit["granted_roles"] = ["role_001", "public"]
    
    client.upsert(
        collection_name="risk_knowledge_base",
        data=results,
        partial_update=True
    )
