from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from .database import Base  # 这行是关键，必须正确导入Base


# 测试用例表
class TestCase(Base):
    __tablename__ = "test_cases"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, comment="用例名称")
    description = Column(Text, comment="用例描述")
    test_type = Column(String(20), default="web", comment="测试类型：web/api")
    url = Column(String(500), comment="测试目标URL")
    steps = Column(Text, comment="测试步骤（JSON格式）")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


# 测试任务表
class TestTask(Base):
    __tablename__ = "test_tasks"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, comment="关联用例ID")
    status = Column(String(20), default="pending", comment="状态：pending/running/success/failed")
    result = Column(Text, comment="测试结果")
    report_path = Column(String(500), comment="Allure报告路径")
    created_at = Column(DateTime, default=datetime.now)
    finished_at = Column(DateTime)