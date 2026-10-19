import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment variables from .env file
load_dotenv()

# Read DATABASE_URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError(
        "DATABASE_URL environment variable is missing. "
        "Please define DATABASE_URL in your .env file (e.g., mysql+pymysql://user:password@localhost:3306/dbname)."
    )

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False,
)

# Create SessionLocal session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Base class for declarative ORM models
Base = declarative_base()


def get_db():
    """
    FastAPI dependency that yields a database session for a request
    and ensures clean closure in a finally block.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
