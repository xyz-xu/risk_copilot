from db.knowledge_dao import KnowledgeDao
from db.sqlite.db_connection import get_conn

class KnowledgeDaoImpl(KnowledgeDao):
    def query_granted_knowledge(self, user_id: str, user_role: str):
        """查询有权限的知识库ID，分为角色权限和用户权限"""
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                    WITH granted_knowledge AS (
                        SELECT DISTINCT collection_name,
                                        partition_name
                        FROM knowledge_user_role_permission
                        WHERE user_id = ? OR
                            user_role = ?
                    )
                    SELECT collection_name,
                        partition_name
                    FROM granted_knowledge
                    WHERE partition_name IS NULL
                    UNION ALL
                    SELECT collection_name,
                        partition_name
                    FROM granted_knowledge t
                    WHERE NOT EXISTS (
                                SELECT 1
                                    FROM granted_knowledge t2
                                    WHERE t.collection_name = t2.collection_name AND
                                        t2.partition_name IS NULL
                            )
                """,
                [user_id, user_role]
            )
            results = cursor.fetchall()
            results = [{"collection_name": row[0], "partition_name": row[1]} for row in results]
            return results
