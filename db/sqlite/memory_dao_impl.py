from db.memory_dao import MemoryDao
from db.sqlite.db_connection import get_conn

class MemoryDaoImpl(MemoryDao):
    def query_long_memory(self, user_id: str):
        """查询给定user_id的长期记忆"""
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "select long_memory from user_long_memory where user_id = ?",
                [user_id]
            )
            results = cursor.fetchall()
            long_memory = results[0][0] if results else ""
            return long_memory if long_memory else ""
        
    def update_long_memory(self, user_id: str, long_memory: str):
        """更新给定user_id的长期记忆"""
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "update user_long_memory set long_memory = ? where user_id = ?",
                [long_memory, user_id]
            )
            return cursor.rowcount

    def select_one_not_sync_checkpointer(self):
        """获取任意一条未被处理的checkpointer"""
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT thread_id
                FROM checkpoints t
                WHERE NOT EXISTS (
                            SELECT 1
                                FROM synced_checkpoints s
                                WHERE t.thread_id = s.thread_id
                        )
                AND thread_id like 'user%'
                LIMIT 1
            """)
            results = cursor.fetchall()
            thread_id = results[0][0] if results else None
            return thread_id

    def insert_synced_checkpointer(self, thread_id: str):
        """新增已同步的checkpoints"""
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "insert into synced_checkpoints(thread_id) values (?)",
                [thread_id]
            )
            return cursor.rowcount
