from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# 使用SQLite文件数据库，无需额外安装
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_platform.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# 数据库依赖，供API接口使用
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()