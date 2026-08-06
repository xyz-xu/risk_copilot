from pymilvus import MilvusClient, utility
import time

# 1. 连接到 Milvus 服务
# 根据你的实际情况，替换 uri 和 token
client = MilvusClient(
    uri="http://localhost:19530",
    db_name="risk_knowledge",
    user="root",
    password=""
)


# 2. 指定你要操作的 Collection 名称
collection_name = "risk_knowledge_base"  # 请替换为你在 Attu 中看到的实际名称

# 3. 执行刷新加载操作
print(f"正在为 Collection '{collection_name}' 执行刷新加载...")
try:
    # refresh_load 方法会尝试将已加载 Collection 中未加载的数据载入内存[citation:5][citation:11]
    utility.refresh_external_collection(collection_name)
    print("刷新加载操作已成功触发。")
except Exception as e:
    print(f"刷新加载操作失败: {e}")
    # 如果失败了，可以在这里添加更详细的错误处理逻辑

# 4. (可选) 检查加载状态
# 刷新操作是异步的，可以稍等几秒再检查状态
time.sleep(2)
load_state = client.get_load_state(collection_name=collection_name)
print(f"Collection '{collection_name}' 的当前加载状态: {load_state}")