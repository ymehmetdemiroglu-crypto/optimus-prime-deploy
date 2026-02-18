from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.core.database import get_db
from .models import User, UserRole
from .schemas import Token, UserCreate, UserUpdate, UserRead
from .dependencies import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    require_admin,
)

router = APIRouter()


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """Authenticate and return a JWT access token."""
    result = await db.execute(select(User).where(User.email == form_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(status_code=400, detail="Account is deactivated")

    user.last_login = datetime.now(timezone.utc)
    await db.commit()

    access_token = create_access_token(
        data={"sub": user.email, "role": user.role.value}
    )
    return Token(access_token=access_token)


@router.get("/me", response_model=UserRead)
async def get_me(current_user: User = Depends(get_current_user)):
    """Return the current authenticated user's profile."""
    return current_user


@router.post("/users", response_model=UserRead)
async def create_user(
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Create a new user (admin only)."""
    result = await db.execute(select(User).where(User.email == user_in.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=user_in.email,
        hashed_password=hash_password(user_in.password),
        full_name=user_in.full_name,
        role=UserRole(user_in.role),
        client_account_id=user_in.client_account_id,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.get("/users", response_model=List[UserRead])
async def list_users(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """List all users (admin only)."""
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    return result.scalars().all()


@router.patch("/users/{user_id}", response_model=UserRead)
async def update_user(
    user_id: int,
    user_in: UserUpdate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Update a user (admin only). Can reset password, toggle active, reassign account."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user_in.full_name is not None:
        user.full_name = user_in.full_name
    if user_in.password is not None:
        user.hashed_password = hash_password(user_in.password)
    if user_in.is_active is not None:
        user.is_active = user_in.is_active
    if user_in.client_account_id is not None:
        user.client_account_id = user_in.client_account_id

    await db.commit()
    await db.refresh(user)
    return user


@router.delete("/users/{user_id}")
async def deactivate_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Deactivate a user (admin only). Does not delete."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = False
    await db.commit()
    return {"status": "success", "message": f"User {user.email} deactivated"}
