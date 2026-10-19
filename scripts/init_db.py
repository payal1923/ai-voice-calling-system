import sys
from pathlib import Path

# Ensure the root directory is in sys.path so app modules can be imported
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from sqlalchemy import inspect
from app.database.database import Base, engine, DATABASE_URL
from app.database.models import (
    SystemUser,
    Customer,
    Call,
    Message,
    Requirement,
    HumanAssistanceRequest,
)


def init_db():
    print("=" * 60)
    print("Database Initialization")
    print("=" * 60)
    print(f"DATABASE_URL target: {DATABASE_URL}")

    models = [
        SystemUser,
        Customer,
        Call,
        Message,
        Requirement,
        HumanAssistanceRequest,
    ]

    print("\nRegistering models with SQLAlchemy Base.metadata:")
    for model in models:
        print(f"  - Model: {model.__name__:<25} -> Table: {model.__tablename__}")

    print("\nCalling Base.metadata.create_all(bind=engine)...")
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    print("\nVerification of Required Tables in Database:")
    all_present = True
    for model in models:
        tbl = model.__tablename__
        if tbl in existing_tables:
            print(f"  [OK] Table '{tbl}' exists.")
        else:
            print(f"  [MISSING] Table '{tbl}' was NOT found.")
            all_present = False

    print("=" * 60)
    if all_present:
        print("Database initialization check complete: All 6 tables verified.")
    else:
        print("Database initialization check complete with missing tables.")
    print("=" * 60)


if __name__ == "__main__":
    init_db()
