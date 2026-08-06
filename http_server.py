from fastapi import FastAPI
from main_agent import get_main_agent
from langgraph.types import Command
from langchain_core.messages import HumanMessage

app = FastAPI()
agent = get_main_agent()

question = "hi"
user_id = "user_001"
user_role = "role_001"

async def handle_user_message(session_id:str, message):
    response = agent.invoke(
        {
            "messages": [HumanMessage(message)],
            "global_context": {
                "question": question,
                "user_id": user_id,
                "user_role": user_role
            }
        },
        config = {"configurable": {"thread_id": user_id + "_" + session_id}}
    )

    if response.get("__interrupt__"):
        return {"status": "completed", "reply": response["__interrupt__"][-1].value}

    return {"status": "completed", "reply": response["messages"][-1].content}

async def handle_interrupt(session_id:str, approved: bool, message: str):
    response = agent.invoke(
        Command(resume={"approved": approved, "checked_list": [message]}),
        config = {"configurable": {"thread_id": user_id + "_" + session_id}}
    )
    return {"status": "completed", "reply": response["messages"][-1].content}

@app.get("/chat")
async def chat(session_id:str, message: str):
    """聊天接口，返回响应"""
    return await handle_user_message(session_id, message)

@app.get("/approve")
async def approve(session_id:str, message: str):
    """审批通过"""
    return await handle_interrupt(session_id, True, message)

@app.get("/")
async def root():
    return {"message": "Hello World"}

# 启动：uvicorn http_server:app --reload
