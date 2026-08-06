
class AuthDao:
    """权限校验接口类"""

    def query_risk_report_agent_permission(self, user_role: str):
        """校验给定的user_role是否有risk_report_agent的权限"""
        return False
    
    def query_role_client_info_permission(self, user_role: str, client_id: str):
        """校验给定的user_role是否有client_id的权限"""
        return False
