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


# 功能测试用例表
class FunctionalTestCase(Base):
    __tablename__ = "functional_test_cases"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(50), comment="用例编号，如TC-ORDER-001")
    name = Column(String(200), nullable=False, comment="用例名称")
    module = Column(String(100), comment="所属模块")
    priority = Column(String(10), comment="优先级：P0/P1/P2/P3")
    preconditions = Column(Text, comment="前置条件")
    test_steps = Column(Text, comment="测试步骤（JSON数组）")
    expected_result = Column(Text, comment="预期结果")
    test_data = Column(Text, comment="测试数据")
    scenario_type = Column(String(50), comment="场景类型")
    source_url = Column(String(500), comment="来源URL")
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