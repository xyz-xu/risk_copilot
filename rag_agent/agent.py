from langchain.tools import tool
from rag_core import rag_service
from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain.agents import create_agent
from llm_core.llm_factory import get_flash_llm
from deepagents import CompiledSubAgent

@tool
def query_granted_permissions():
    """查询当前在知识库里，有哪些collection_name的权限"""
    return "被授予的权限有：risk_knowledge_base"

@tool
def query_knowledge(collection_name: str, question: str):
    """使用稠密向量检索（语义检索），在个人知识库中搜索相关信息。搜索时使用完整的问题或关键短语。

    Args:
        collection_name: collection_name，查询所在的集合
        question: 搜索问题或关键短语
    """
    if not collection_name:
        return "查询失败，collection_name不可为空"
    if not question:
        return "查询失败，搜索问题或关键短语，不可为空"
    
    llm_friendly_docs = rag_service.rag_query(question, collection_name)
    content = "\n".join(f"- 评分：{score}，内容：{doc["entity"]["text"]}" for score, doc in llm_friendly_docs)

    return f"知识库查询结果：\n{content}"

@tool
def query_knowledge_using_sparse(collection_name: str, question: str):
    """使用全文检索，在个人知识库中搜索相关信息。搜索时使用完整的问题或关键短语。

    Args:
        collection_name: collection_name，查询所在的集合
        question: 搜索问题或关键短语
    """
    if not collection_name:
        return "查询失败，collection_name不可为空"
    if not question:
        return "查询失败，搜索问题或关键短语，不可为空"
    
    llm_friendly_docs = rag_service.sparse_search(question, collection_name)
    content = "\n".join(f"- 评分：{score}，内容：{doc["entity"]["text"]}" for score, doc in llm_friendly_docs)

    return f"知识库查询结果：\n{content}"

@tool
def query_knowledge_using_hybrid(collection_name: str, question: str):
    """使用混合检索，在个人知识库中搜索相关信息。搜索时使用完整的问题或关键短语。

    Args:
        collection_name: collection_name，查询所在的集合
        question: 搜索问题或关键短语
    """
    if not collection_name:
        return "查询失败，collection_name不可为空"
    if not question:
        return "查询失败，搜索问题或关键短语，不可为空"
    
    llm_friendly_docs = rag_service.hybrid_search(question, collection_name)
    content = "\n".join(f"- 评分：{score}，内容：{doc["entity"]["text"]}" for score, doc in llm_friendly_docs)

    return f"知识库查询结果：\n{content}"

@tool
def query_knowledge_using_graph(question: str):
    """使用知识图谱检索，在个人知识库中搜索相关信息。搜索时使用完整的问题或关键短语。

    Args:
        question: 搜索问题或关键短语
    """
    llm_friendly_docs = rag_service.graph_query(question)
    content = "\n".join(f"- 评分：{score}，内容：{doc["text"]}" for score, doc in llm_friendly_docs)

    return f"知识库查询结果：\n{content}"

def get_rag_agent(parent_checkpointer: BaseCheckpointSaver = None):
    agent = create_agent(
        model=get_flash_llm(),
        tools=[query_granted_permissions, query_knowledge, query_knowledge_using_sparse, query_knowledge_using_hybrid, query_knowledge_using_graph],
        system_prompt="""你是个人知识库助手。
                        ## 规则
                        1. 所有问题必须先用 query_granted_permissions 工具检索有权限的知识库，即collection_name
                        2. 再选择合适的检索方式去查询知识库
                        3. 如果知识库中没有相关内容，如实告知
                        4. 回答要结构化，使用数字列表或分段""",
        checkpointer=parent_checkpointer if parent_checkpointer else Non
    )

    complied_sub_agent = CompiledSubAgent(
        name="rag_subagent",
        description="用于查询知识库的Agent",
        runnable=agent
    )
    return complied_sub_agent
