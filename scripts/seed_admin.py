import sys
from pathlib import Path

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from datetime import datetime, timezone
from app.database.database import SessionLocal
from app.database.models import SystemUser
from app.utils.auth import hash_password


def seed_admin_user():
    """
    Idempotent script to seed initial admin user if not present.
    """
    db = SessionLocal()
    try:
        existing_admin = (
            db.query(SystemUser)
            .filter(SystemUser.username == "admin")
            .first()
        )
        if existing_admin:
            print(f"[INFO] System user 'admin' already exists (ID: {existing_admin.id}, Role: {existing_admin.role}). Skipping creation.")
            return

        now = datetime.now(timezone.utc)
        admin_user = SystemUser(
            username="admin",
            password_hash=hash_password("AdminPass123!"),
            role="admin",
            created_at=now,
            updated_at=now,
        )
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        print(f"[SUCCESS] Admin user created successfully (ID: {admin_user.id}, Username: {admin_user.username}, Role: {admin_user.role}).")
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Failed to seed admin user: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_admin_user()
