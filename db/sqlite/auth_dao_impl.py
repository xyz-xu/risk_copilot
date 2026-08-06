from db.auth_dao import AuthDao
from db.sqlite.db_connection import get_conn

class AuthDaoImpl(AuthDao):
    def query_risk_report_agent_permission(self, user_role: str):
        """校验给定的user_role是否有risk_report_agent的权限"""
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "select 1 from risk_report_agent_permission where user_role = ?",
                [user_role]
            )
            results = cursor.fetchall()
            return len(results) > 0
    
    def query_role_client_info_permission(self, user_role: str, client_id: str):
        """校验给定的user_role是否有client_id的权限"""
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "select 1 from role_client_info_permission where user_role = ? and client_id = ?",
                [user_role, client_id]
            )
            results = cursor.fetchall()
            return len(results) > 0
