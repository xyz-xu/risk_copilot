from deepagents import create_deep_agent
from risk_report_agent.agent import get_risk_report_subagent
from rag_agent.agent import get_rag_agent
from db.dao_impl import checkpointer
from llm_core.llm_factory import get_flash_llm
from common.states import RiskDeepAgentState
from langchain.tools import tool

@tool
def query_granted_permissions():
    """查询当前被授予的权限"""
    return "被授予的权限有：新增，删除，修改，查询"

def get_main_agent():
    risk_report_subagent = get_risk_report_subagent(checkpointer)
    rag_subagent = get_rag_agent(checkpointer)

    main_agent = create_deep_agent(
        model=get_flash_llm(),
        tools=[query_granted_permissions],
        subagents=[risk_report_subagent, rag_subagent],
        checkpointer=checkpointer,
        state_schema=RiskDeepAgentState,
        system_prompt="""
            你是一位风控助手，拥有处理问题，和拆分分体给subagent处理的能力。
            特别的，针对生成风控报告的任务，直接交给risk_report_subagent处理，且不需要修改其返回值。
            如果问题是简单的查询知识库，可调用rag_subagent。
        """
    )
    return main_agent
