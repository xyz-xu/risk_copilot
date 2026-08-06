from db.dao_impl import memory_dao, checkpointer
from llm_core.llm_factory import get_flash_llm
from langchain_core.messages import HumanMessage
import time

def start():
    while True:
        thread_id = memory_dao.select_one_not_sync_checkpointer()
        if thread_id:
            user_id = thread_id[0:8]
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 开始处理{thread_id}-{user_id}")

            long_memory = memory_dao.query_long_memory(user_id)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 获取长期记忆：{long_memory}")

            checkpoints = checkpointer.list({"configurable": {"thread_id": thread_id}})

            messages = []
            for _, _, _, _, lists in checkpoints:
                for _, type, values in lists:
                    if type == "messages":
                        messages = messages + [value for value in values if isinstance(value, HumanMessage)]
            messages = messages[::-1]
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 载入历史对话记录{len(messages)}条")

            if messages:
                # messages非空，则生成新的长期记忆
                model = get_flash_llm()
                response = model.invoke([
                    {
                        "role": "system",
                        "content": """
                                # 角色
                                你擅长总结长期记忆，请根据已有的长期记忆，和上下文对话，总结新的长期记忆。
                                长期记忆包含用户明确的偏好、习惯和长期目标。
                                # 返回示例
                                喜欢称呼为学长，喜欢在晚上使用
                                """
                    },
                    HumanMessage(f"历史长期记忆：{long_memory}"),
                    *messages
                ])

                new_long_memory = response.content
                cnt = memory_dao.update_long_memory(user_id, new_long_memory)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {cnt}更新{user_id}长期记忆为：{new_long_memory}")
            
            memory_dao.insert_synced_checkpointer(thread_id)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 结束处理{thread_id}-{user_id}")
        else:
            time.sleep(10)  # 每10秒执行一次
