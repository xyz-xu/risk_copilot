from common.exceptions import AgentPermissionException
from common.states import RiskMessagesState

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

from db.dao_impl import *
from rag_core import rag_service
from common.id_utils import gen_tool_id
from llm_core.llm_factory import get_flash_llm, get_pro_llm
from common.structured_output import ClientName, ReportObservatioin
from langgraph.types import interrupt, Command
from langgraph.checkpoint.base import BaseCheckpointSaver
from deepagents import CompiledSubAgent

# 合并global_context
def merge_blobal_context_node(state: RiskMessagesState):
    """将global_context的值合并至state"""
    return state.get("global_context", {})

# 权限校验
def authentication_node(state: RiskMessagesState):
    """判断登录角色是否有权限访问risk_report_agent"""
    user_role = state.get("user_role", "guest_role")
    check = auth_dao.query_risk_report_agent_permission(user_role=user_role)
    if not check:
        raise AgentPermissionException(f"当前角色{user_role}没有访问报表Agent的权限")
    return {"role_permission": True}

# 加载长期记忆
def load_long_memory_node(state: RiskMessagesState):
    """从数据库中加载长期记忆"""
    if state.get("load_long_memory", False):
        # 无需重复加载
        return {}
    
    user_id = state.get("user_id", "guest_user")
    long_memory = memory_dao.query_long_memory(user_id=user_id)
    message = f"已加载长期记忆：{long_memory if long_memory else "空"}"
    tool_call_id = gen_tool_id()
    return {
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [{"id": tool_call_id, "name": "query_long_memory", "args": {}}]},
            {"role": "tool", "content": message, "tool_call_id": tool_call_id}
        ],
        "load_long_memory": True
    }

# 加载知识库
def load_knowledge_node(state: RiskMessagesState):
    """从知识库加载相关知识"""
    if state.get("load_knowledge", False):
        # 无需重复加载
        return {}
    
    user_id = state.get("user_id", "guest_user")
    user_role = state.get("user_role", "guest_role")
    
    # 查询有权限的知识库
    knowledges = knowledge_dao.query_granted_knowledge(user_id=user_id, user_role=user_role)

    # 加载相关知识
    question = state.get("question")
    contents = []
    for knowledge in knowledges:
        llm_friendly_docs = rag_service.rag_query(question, knowledge["collection_name"], knowledge["partition_name"], user_role)
        content = "\n".join(f"- 评分：{score}，内容：{doc["entity"]["text"]}" for score, doc in llm_friendly_docs)
        contents.append(content)
    
    message = f"已加载知识库：{"\n".join(contents) if contents else "空"}"
    tool_call_id = gen_tool_id()
    return {
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [{"id": tool_call_id, "name": "rag_query", "args": {}}]},
            {"role": "tool", "content": message, "tool_call_id": tool_call_id}
        ],
        "load_knowledge": True
    }

# 解析客户名
def find_client_name_node(state: RiskMessagesState):
    """调用大模型，解析客户名，用于后续权限校验"""
    structured_model = get_flash_llm().with_structured_output(ClientName)

    message = {"role": "system", "content": "请从上下文中解析出客户名称，并返回客户名称列表，若为解析到，则返回空"}
    human_massages = [msg for msg in state.get("messages") if isinstance(msg, HumanMessage)]
    res = structured_model.invoke([message, *human_massages])
    client_list = res.names if res else []

    global_context=state.get("global_context",{})
    global_context["report_client_list"]= client_list
    return {"report_client_list": client_list, "global_context": global_context}

# hitl 确认是否查询给定客户的信息
def hitl_check_query_client_info_node(state: RiskMessagesState):
    """敏感操作，返回确认"""
    client_list = state.get("report_client_list", [])
    if not client_list:
        raise AgentPermissionException(f"请指定具体的客户姓名")

    command = interrupt(f"敏感操作，请确认生成{client_list}的风控报告")
    if not command["approved"]:
        raise AgentPermissionException(f"请确认生成")
    
    return {"commands": [command]}

# 校验客户权限
def check_role_client_info_permission_node(state: RiskMessagesState):
    """校验角色是否有给定客户的权限"""
    user_role = state.get("user_role", "guest_role")
    client_list = state.get("report_client_list", [])

    for client_id in client_list:
        check = auth_dao.query_role_client_info_permission(user_role=user_role, client_id=client_id)
        if not check:
            raise AgentPermissionException(f"角色{user_role}没有客户{client_id}的查询权限")
    
    return {}

# 查询客户风控基本信息
def query_client_risk_info_node(state: RiskMessagesState):
    """查询给定客户的风控基本信息"""
    client_list = state.get("report_client_list", [])

    messages = []
    for client_id in client_list:
        info = client_dao.query_client_risk_info(client_id)
        message = f"客户:{client_id}\n风控信息：{info if info else "空"}"
        tool_call_id = gen_tool_id()
        messages.append({"role": "assistant", "content": "", "tool_calls": [{"id": tool_call_id, "name": "rag_query", "args": {}}]})
        messages.append({"role": "tool", "content": message, "tool_call_id": tool_call_id})
    
    return {"messages": messages}

# reasoning
def llm_reasoning_node(state: RiskMessagesState):
    """使用pro模型来进行推理，不需要具体行动"""
    model = get_pro_llm()

    message = model.invoke([
        {
            "role": "system",
            "content": """
                    你是一位专业的风控专家，现在有个生成风控报告的任务，请你给出详细的推理过程，不需要行动
                       """
        },
        *state.get("messages")
    ])
    return {"messages": [message], "risk_report_loop_cnt": state.get("risk_report_loop_cnt", 0)+1}

# acting
def llm_acting_node(state: RiskMessagesState):
    """使用flash模型来进行任务"""
    model = get_flash_llm()

    message = model.invoke([
        {
            "role": "system",
            "content": """
                    你是一位专业的风控专家，现在有个生成风控报告的任务，请你依据之前的推理，生成详细的风控报告，报告使用markdown语法
                       """
        },
        *state.get("messages")
    ])
    
        
    global_context=state.get("global_context",{})
    global_context["risk_report"]= message.content
    return {
        "messages": [message],
        "risk_report": message.content,
        "global_context": global_context
    }

# observation
def llm_observation_node(state: RiskMessagesState):
    """使用pro模型来检查"""
    model = get_pro_llm().with_structured_output(ReportObservatioin)

    result = model.invoke([
        {
            "role": "system",
            "content": """
                    你是一位专业的风控专家，现在有个已经生成好的风控报告，请判断是否符合你之前的推理的要求。
                    若通过请给出True，否则请给出False。同时请附上具体的意见。
                       """
        },
        *state.get("messages")
    ])
    return {"risk_observation": result}

# observation router
def observation_router(state: RiskMessagesState):
    """依据观察结果，决定时重做还是结束，循环到第三次则直接结束"""
    risk_report_loop_cnt = state.get("risk_report_loop_cnt", 0)

    if state["risk_observation"].approved or risk_report_loop_cnt >= 3:
        return "go_end"
    
    state["messages"].append(AIMessage(content=f"请参考修改意见：{state["risk_observation"].opinion}"))
    return "go_reasoning"

def gen_graph(checkpointer: BaseCheckpointSaver):
    # 创建builder
    builder = StateGraph(RiskMessagesState)

    # 创建node
    builder.add_node("merge_blobal_context_node", merge_blobal_context_node)
    builder.add_node("authentication_node", authentication_node)
    builder.add_node("load_long_memory_node", load_long_memory_node)
    builder.add_node("load_knowledge_node", load_knowledge_node)
    builder.add_node("find_client_name_node", find_client_name_node)
    builder.add_node("hitl_check_query_client_info_node", hitl_check_query_client_info_node)
    builder.add_node("check_role_client_info_permission_node", check_role_client_info_permission_node)
    builder.add_node("query_client_risk_info_node", query_client_risk_info_node)
    builder.add_node("llm_reasoning_node", llm_reasoning_node)
    builder.add_node("llm_acting_node", llm_acting_node)
    builder.add_node("llm_observation_node", llm_observation_node)

    # 创建edge
    builder.add_edge("merge_blobal_context_node", "authentication_node")
    builder.add_edge("authentication_node", "load_long_memory_node")
    builder.add_edge("load_long_memory_node", "load_knowledge_node")
    builder.add_edge("load_knowledge_node", "find_client_name_node")
    builder.add_edge("find_client_name_node", "hitl_check_query_client_info_node")
    builder.add_edge("hitl_check_query_client_info_node", "check_role_client_info_permission_node")
    builder.add_edge("check_role_client_info_permission_node", "query_client_risk_info_node")
    builder.add_edge("query_client_risk_info_node", "llm_reasoning_node")
    builder.add_edge("llm_reasoning_node", "llm_acting_node")
    builder.add_edge("llm_acting_node", "llm_observation_node")

    # condition END
    builder.add_conditional_edges(
        "llm_observation_node",
        observation_router,
        {
            "go_reasoning": "llm_reasoning_node",
            "go_end": END
        }
    )

    # START
    builder.add_edge(START, "merge_blobal_context_node")

    # complie
    graph = builder.compile(checkpointer=checkpointer)
    return graph

def get_risk_report_subagent(parent_checkpointer: BaseCheckpointSaver = None):
    db_saver = parent_checkpointer if parent_checkpointer else checkpointer

    risk_report_graph = gen_graph(db_saver)

    subagent = CompiledSubAgent(
        name="risk_report_subagent",
        description="风控专家subagent，用于生成风控报告",
        runnable=risk_report_graph
    )

    return subagent

"""
def _test():
    graph = gen_graph(checkpointer)
    # ask
    thread_id = gen_tool_id()
    response = graph.invoke(
        {
            "messages": [HumanMessage("请生成张三和李四的风控报告")],
            "global_context": {
                "question": "hi",
                "user_id": "user_001",
                "user_role": "role_001"
            }
        },
        config = {"configurable": {"thread_id": thread_id}}
    )
    # print(response)

    # '__interrupt__': [Interrupt(value="请确认生成['张三', '李四']的风控报告", id='ced10eef61883089d3f8a1477f0eae62')]
    # print(response["__interrupt__"])

    # continue
    response = graph.invoke(
        Command(resume={"approved": True, "checked_list": response["report_client_list"]}),
        config = {"configurable": {"thread_id": thread_id}}
    )
    print(response)
"""
