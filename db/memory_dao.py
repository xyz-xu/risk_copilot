class MemoryDao:
    """记忆接口类"""

    def query_long_memory(self, user_id: str):
        """查询给定user_id的长期记忆"""
        return "";

    def update_long_memory(self, user_id: str, long_memory: str):
        """更新给定user_id的长期记忆"""
        pass

    def select_one_not_sync_checkpointer(self):
        """获取任意一条未被处理的checkpointer"""
        pass

    def insert_synced_checkpointer(self, thread_id: str):
        """新增已同步的checkpoints"""
        pass
