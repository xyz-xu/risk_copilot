from deepagents import DeepAgentState
from langgraph.graph.message import MessagesState

class RiskMessagesState(MessagesState):
    """拓展MessageState，用于在Agent和Graph间流转"""
    global_context: dict
    user_id: str
    user_role: str
    role_permission: bool
    load_long_memory: bool
    question: str
    load_knowledge: bool
    report_client_list: list[str]
    report_client_approved: bool
    commands: list[dict]
    risk_report: str
    risk_observation: dict
    risk_report_loop_cnt: int

class RiskDeepAgentState(DeepAgentState):
    """拓展DeepAgentState"""
    global_context: dict
