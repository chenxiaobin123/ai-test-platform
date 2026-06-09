from pydantic import BaseModel
from datetime import datetime
from typing import Optional


# 测试用例相关
class TestCaseCreate(BaseModel):
    name: str
    description: Optional[str] = None
    test_type: str = "web"
    url: str
    steps: str  # JSON字符串格式的步骤


class TestCaseResponse(TestCaseCreate):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# 功能测试用例相关
class FunctionalTestCaseCreate(BaseModel):
    case_id: str = ""
    name: str
    module: str = ""
    priority: str = "P1"
    preconditions: str = ""
    test_steps: str = "[]"
    expected_result: str = ""
    test_data: str = ""
    scenario_type: str = ""
    source_url: str = ""


class FunctionalTestCaseResponse(FunctionalTestCaseCreate):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# 测试任务相关
class TestTaskResponse(BaseModel):
    id: int
    case_id: int
    status: str
    result: Optional[str]
    report_path: Optional[str]
    created_at: datetime
    finished_at: Optional[datetime]

    class Config:
        from_attributes = True