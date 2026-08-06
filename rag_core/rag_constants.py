milvus_uri="http://localhost:19530"
milvus_db_name="risk_knowledge"
milvus_user="root"
milvus_password=""

milvus_anns_field = "embedding"
milvus_sparse_anns_field = "sparse"

# radius可直接过滤上限参数（向量检索）
milvus_filtered_radius = 0.3

milvus_limit = 20

milvus_dense_weight = 1.0
milvus_sparse_weight = 0.7

reranker_limit = 5

embedding_dim = 1024