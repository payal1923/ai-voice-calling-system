import uuid
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database.models import SystemUser
from app.utils.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    require_admin,
)

router = APIRouter()

# In-memory sliding rate limiter: ip -> list of timestamps
_caller_session_requests = defaultdict(list)
_RATE_LIMIT_MAX_REQUESTS = 15
_RATE_LIMIT_WINDOW_SECONDS = 60


def _check_caller_rate_limit(client_ip: str):
    now = time.time()
    valid_timestamps = [t for t in _caller_session_requests[client_ip] if now - t < _RATE_LIMIT_WINDOW_SECONDS]
    if len(valid_timestamps) >= _RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many caller session requests. Please wait before connecting again.",
        )
    valid_timestamps.append(now)
    _caller_session_requests[client_ip] = valid_timestamps


class CallerSessionRequest(BaseModel):
    client_type: Optional[str] = "web"
    caller_name: Optional[str] = "Guest Caller"
    device_id: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "system_user"


class UserResponse(BaseModel):
    id: int
    username: str
    role: str
    created_at: Optional[str] = None


@router.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate system user and return JWT access token."""
    username = req.username.strip()
    user = db.query(SystemUser).filter(SystemUser.username == username).first()

    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = create_access_token(data={"sub": user.username, "role": user.role, "uid": user.id})

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role,
        },
    }


@router.post("/caller-session")
def create_caller_session(req: Optional[CallerSessionRequest] = None, request: Request = None):
    """
    Creates a temporary, scoped JWT session for public website visitors or mobile app callers.
    Protected with IP-based rate limiting to prevent resource exhaustion on local AI inference.
    """
    client_ip = "127.0.0.1"
    if request:
        client_ip = (
            request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else "127.0.0.1")
        )
    _check_caller_rate_limit(client_ip)

    caller_id = f"caller_{uuid.uuid4().hex[:12]}"
    client_type = (req.client_type if req and req.client_type else "web").strip().lower()
    caller_name = (req.caller_name if req and req.caller_name else "Guest Caller").strip()

    token = create_access_token(
        data={
            "sub": caller_id,
            "role": "caller",
            "client_type": client_type,
            "caller_name": caller_name,
        },
        expires_delta=timedelta(hours=4),
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "caller_id": caller_id,
        "role": "caller",
        "client_type": client_type,
    }


@router.post("/logout")
def logout(current_user: SystemUser = Depends(get_current_user)):
    """Logs out current user session."""
    return {"status": "logged_out", "username": current_user.username}


@router.get("/me")
def get_me(current_user: SystemUser = Depends(get_current_user)):
    """Returns profile information for the authenticated user."""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "role": current_user.role,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
    }


@router.get("/users")
def list_system_users(
    admin: SystemUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only endpoint to list all registered system users."""
    users = db.query(SystemUser).order_by(SystemUser.id.asc()).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "role": u.role,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "updated_at": u.updated_at.isoformat() if u.updated_at else None,
        }
        for u in users
    ]


@router.post("/users")
def create_system_user(
    req: CreateUserRequest,
    admin: SystemUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only endpoint to create a new system user."""
    username = req.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    if not req.password or len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if req.role not in ["admin", "system_user"]:
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'system_user'")

    existing = db.query(SystemUser).filter(SystemUser.username == username).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"User '{username}' already exists")

    now = datetime.now(timezone.utc)
    new_user = SystemUser(
        username=username,
        password_hash=hash_password(req.password),
        role=req.role,
        created_at=now,
        updated_at=now,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "id": new_user.id,
        "username": new_user.username,
        "role": new_user.role,
        "created_at": new_user.created_at.isoformat(),
    }


@router.delete("/users/{user_id}")
def delete_system_user(
    user_id: int,
    admin: SystemUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Admin-only endpoint to delete a system user."""
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete your own admin account")

    target = db.query(SystemUser).filter(SystemUser.id == user_id).first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(target)
    db.commit()
    return {"status": "deleted", "user_id": user_id}
