from pydantic import BaseModel, Field
from typing import List


class ClientName(BaseModel):
    """从文本中提取人物姓名信息"""
    names: List[str] = Field(description="人物姓名列表")

class ReportObservatioin(BaseModel):
    """针对任务的具体观察结果"""
    approved: bool = Field(description="生成的风控报告是否满足要求")
    opinion: str = Field(description="具体的意见")