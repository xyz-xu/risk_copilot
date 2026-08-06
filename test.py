def test1():
    from risk_report_agent import agent
    agent._test()

def test2():
    from main_agent import get_main_agent
    from langgraph.types import Command
    from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
    import uuid
    agent = get_main_agent()
    thread_id = str(uuid.uuid4())
    response = agent.invoke(
        {
            "messages": [HumanMessage("生成张三的风控报告")],
            "global_context": {
                "question": "hi",
                "user_id": "user_001",
                "user_role": "role_001"
            }
        },
        config = {"configurable": {"thread_id": thread_id}}
    )
    print("#1" + '-'*50)
    print(response)

    print("#2" + '-'*50)
    print(response["__interrupt__"])

    # continue
    print("#3" + '-'*50)
    response = agent.invoke(
        Command(resume={"approved": True, "checked_list": response["global_context"]["report_client_list"]}),
        config = {"configurable": {"thread_id": thread_id}}
    )
    print(response)

def test3():
    from rag_core import rag_service
    llm_friendly_docs = rag_service.sparse_search("风控", "risk_knowledge_base")
    content = "\n".join(f"- 评分：{score}，内容：{doc["entity"]["text"]}" for score, doc in llm_friendly_docs)
    print(content)

def test4():
    from rag_core import rag_service
    llm_friendly_docs = rag_service.hybrid_search("风控", "risk_knowledge_base")
    content = "\n".join(f"- 评分：{score}，内容：{doc["entity"]["text"]}" for score, doc in llm_friendly_docs)
    print(content)

def test5():
    from rag_core.graph_rag import text_chunk_proc
    text_chunk_proc.run()

def test6():
    from rag_core.graph_rag import text_persist
    a = text_persist.load(text_persist.type_relationid_2_passageids)
    print(a)

def test7():
    from rag_core.graph_rag import  graph_adj_matrix
    graph_adj_matrix.load_adj()

def test8():
    from rag_core import rag_service
    rag_service.graph_query("")

test8()
