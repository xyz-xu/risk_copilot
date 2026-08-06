from db.client_dao import ClientDao
from db.sqlite.db_connection import get_conn

class ClientDaoImpl(ClientDao):
    def query_client_risk_info(self, client_id: str):
        """查询给定客户的风控信息"""
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "select risk_info from client_info where client_id = ?",
                [client_id]
            )
            results = cursor.fetchall()
            return results[0][0] if results else ""
