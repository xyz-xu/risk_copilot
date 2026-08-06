from langchain.chat_models import init_chat_model
import dotenv

dotenv.load_dotenv()

_flash_llm = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=1,
    extra_body={"thinking": {"type": "disabled"}}
    )

_pro_llm = init_chat_model(
    "deepseek:deepseek-v4-pro",
    temperature=1,
    extra_body={"thinking": {"type": "disabled"}}
    )

_flash_llm_with_thinking = init_chat_model(
    "deepseek:deepseek-v4-flash",
    temperature=1
    )

_pro_llm_with_thinking = init_chat_model(
    "deepseek:deepseek-v4-pro",
    temperature=1
    )

def get_flash_llm():
    return _flash_llm;

def get_pro_llm():
    return _pro_llm;
