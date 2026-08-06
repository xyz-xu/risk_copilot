from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from rag_core.rag_constants import embedding_dim
from pymilvus import (
    Function,
    FunctionType,
    MilvusClient,
    DataType,
    CollectionSchema,
    FieldSchema
)
from rag_core.rag_factory import get_milvus_client, get_embedding_model
from tqdm import tqdm
import time
import uuid

def load_text_lines(filename: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " "],
    )

    loader = TextLoader(filename, encoding="utf-8")
    docs = loader.load()

    chunks = text_splitter.split_documents(docs)

    text_lines = [chunk.page_content for chunk in chunks]
    return text_lines

def load_markdown_lines(filename: str):
    markdown_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=500, chunk_overlap=200
    )

    loader = TextLoader(filename, encoding="utf-8")
    docs = loader.load()

    chunks = markdown_splitter.split_documents(docs)

    text_lines = [chunk.page_content for chunk in chunks]
    return text_lines

def load_lines():
    filename = "resources/risk_info.md"

    if filename.lower().endswith(".md"):
        return load_markdown_lines(filename)
    else:
        load_text_lines(filename)

# schema
def create_collection_schema():
    fields = [
        # 主键字段
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,  # 手动指定 ID
            description="文档唯一ID"
        ),
        # 原文存储字段（设置足够长度）
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535,
            description="文档原文内容",
            enable_analyzer=True
        ),
        # 向量字段（核心检索字段）
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=1024,  # 根据你的 Embedding 模型调整，如 text-embedding-ada-002 是 1536
            description="文本向量"
        ),
        # 稀疏向量字段（BM25全文检索）
        FieldSchema(
            name="sparse",
            dtype=DataType.SPARSE_FLOAT_VECTOR,
            description="稀疏向量"
        ),
        # 权限控制字段（数组类型）
        FieldSchema(
            name="granted_roles",
            dtype=DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_length=50,
            max_capacity=20,  # 最多允许 20 个角色有权限
            description="允许访问此文档的角色列表"
        ),
        # 文档类型（用于辅助过滤）
        FieldSchema(
            name="doc_type",
            dtype=DataType.VARCHAR,
            max_length=64,
            description="文档类型，如 policy, case, report"
        ),
        # 文档原文来源（全局唯一，用于文档更新；范围尽可能小）
        FieldSchema(
            name="source_uri",
            dtype=DataType.VARCHAR,
            max_length=500,
            description="文档原文来源（全局唯一，用于文档更新；范围尽可能小）"
        ),
        # 创建时间
        FieldSchema(
            name="create_time",
            dtype=DataType.INT64,
            description="创建时间戳"
        )
    ]

    bm25_function = Function(
        name="text_bm25_emb", # Function name
        input_field_names=["text"], # Name of the VARCHAR or TEXT field containing raw text data
        output_field_names=["sparse"], # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
        function_type=FunctionType.BM25, # Set to `BM25`
    )
    functions = [bm25_function]
    
    return CollectionSchema(
        fields=fields,
        description="风控知识库文档集合",
        functions=functions,
        enable_dynamic_field=False  # 建议关闭动态字段以保持性能
    )

# collection
def create_collection(client: MilvusClient, collection_name: str):
    # 如果集合已存在，先删除（谨慎操作）
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    
    schema = create_collection_schema()
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        # 可选：设置分片数（数据量大时可调整）
        # num_shards=2
    )

    # 3. 创建索引（提升查询性能）
    index_params = client.prepare_index_params()

    # 稠密向量索引；向量检索
    index_params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",  # 或 "HNSW"，根据你的精度/性能要求选择；IVF_FLAT内存占用小，HNSW搜索质量更高但内存占用大
        metric_type="IP",   # 语义检索首选余弦相似度
        params={"nlist": 128}   # 簇的数量，建议设为 sqrt(向量数) 左右，如 100万条数据设 1000-2000
    )

    # BM25索引；全文检索
    index_params.add_index(
        field_name="sparse", # Name of the sparse vector field to index
        index_type="SPARSE_INVERTED_INDEX", # Type of the index to create
        index_name="sparse_bm25_index", # Name of the index to create
        metric_type="BM25", # Metric type used for full text search
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )
    
    client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )
    
    print(f"Collection '{collection_name}' 创建成功！")

def init_collection_and_partition():
    client = get_milvus_client()

    # 创建集合
    collection_name = "risk_knowledge_base"
    create_collection(client, collection_name)

    # 创建分区
    partition_name = "partition_001"
    client.create_partition(
        collection_name=collection_name,
        partition_name=partition_name
    )

def run(init: bool = False):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START")

    # 初始化集合
    if init:
        init_collection_and_partition()

    # 新增数据
    doc_lines = load_lines()
    embedding_model = get_embedding_model()
    milliseconds = int(time.time() * 1000)

    doc_embeddings = embedding_model.embed_documents(doc_lines)

    data_list = []
    for i, line in enumerate(tqdm(doc_lines, desc="Creating embeddings")):
        data = {
            "text": line,
            "embedding": doc_embeddings[i],  # 你的向量
            "granted_roles": ["role_001", "public"],
            "doc_type": "policy",
            "source_uri": "resources/risk_info.md",
            "create_time": milliseconds
        }
        data_list.append(data)

    client = get_milvus_client()
    
    collection_name = "risk_knowledge_base"
    partition_name = "partition_001"
    client.insert(collection_name=collection_name, data=data_list, partition_name=partition_name)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] END")