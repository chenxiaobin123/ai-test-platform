from sqlalchemy.orm import Session
from . import models, schemas
from datetime import datetime

# 创建测试用例
def create_test_case(db: Session, case: schemas.TestCaseCreate):
    db_case = models.TestCase(**case.model_dump())  # 修复：dict() → model_dump()
    db.add(db_case)
    db.commit()
    db.refresh(db_case)
    return db_case

# 获取所有测试用例
def get_test_cases(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.TestCase).offset(skip).limit(limit).all()

# 获取单个测试用例
def get_test_case(db: Session, case_id: int):
    return db.query(models.TestCase).filter(models.TestCase.id == case_id).first()

# 创建测试任务
def create_test_task(db: Session, case_id: int):
    db_task = models.TestTask(case_id=case_id)
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

# 更新测试任务状态
def update_test_task(db: Session, task_id: int, status: str, result: str = None, report_path: str = None):
    db_task = db.query(models.TestTask).filter(models.TestTask.id == task_id).first()
    if db_task:
        db_task.status = status
        db_task.result = result
        db_task.report_path = report_path
        db_task.finished_at = datetime.now()
        db.commit()
        db.refresh(db_task)
    return db_task

# 获取测试任务
def get_test_task(db: Session, task_id: int):
    return db.query(models.TestTask).filter(models.TestTask.id == task_id).first()

# 新增：更新测试用例
# 更新测试用例（兼容Pydantic模型和dict）
def update_test_case(db: Session, case_id: int, case):
    db_case = db.query(models.TestCase).filter(models.TestCase.id == case_id).first()
    if not db_case:
        return None

    # 兼容 Pydantic v1/v2
    if hasattr(case, "model_dump"):
        case_data = case.model_dump()
    elif hasattr(case, "dict"):
        case_data = case.dict()
    else:
        case_data = case

    for key, value in case_data.items():
        if hasattr(db_case, key):
            setattr(db_case, key, value)

    db.commit()
    db.refresh(db_case)
    return db_case

# 新增：删除测试用例
def delete_test_case(db: Session, case_id: int):
    db_case = db.query(models.TestCase).filter(models.TestCase.id == case_id).delete()
    db.commit()
    return db_case

#新增批量查询函数
def get_test_tasks_by_ids(db: Session, task_ids: list[int]):
    return db.query(models.TestTask).filter(models.TestTask.id.in_(task_ids)).all()


# 功能测试用例CRUD
def create_functional_case(db: Session, case: schemas.FunctionalTestCaseCreate):
    db_case = models.FunctionalTestCase(**case.model_dump())
    db.add(db_case)
    db.commit()
    db.refresh(db_case)
    return db_case


def get_functional_cases(db: Session, skip: int = 0, limit: int = 200):
    return db.query(models.FunctionalTestCase).order_by(models.FunctionalTestCase.id.desc()).offset(skip).limit(limit).all()


def get_functional_case(db: Session, case_id: int):
    return db.query(models.FunctionalTestCase).filter(models.FunctionalTestCase.id == case_id).first()


def update_functional_case(db: Session, case_id: int, case):
    db_case = db.query(models.FunctionalTestCase).filter(models.FunctionalTestCase.id == case_id).first()
    if not db_case:
        return None
    if hasattr(case, "model_dump"):
        case_data = case.model_dump()
    elif hasattr(case, "dict"):
        case_data = case.dict()
    else:
        case_data = case
    for key, value in case_data.items():
        if hasattr(db_case, key):
            setattr(db_case, key, value)
    db.commit()
    db.refresh(db_case)
    return db_case


def delete_functional_case(db: Session, case_id: int):
    db_case = db.query(models.FunctionalTestCase).filter(models.FunctionalTestCase.id == case_id).delete()
    db.commit()
    return db_case