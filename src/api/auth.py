from datetime import datetime, timedelta
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

from src.config.settings import settings
from src.models import TokenRequest, TokenResponse  # Updated import

router = APIRouter()

# Security utilities
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# JWT settings
SECRET_KEY = settings.api_key  # Not recommended for production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Mock user database (replace with actual database in production)
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("password"),
        "is_active": True,
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify the password against the hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_user(username: str) -> Optional[Dict]:
    """Get user from the database."""
    if username in USERS_DB:
        return USERS_DB[username]
    return None


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """Authenticate a user."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    """
    Authenticate and get an access token.

    Args:
        form_data: OAuth2 password request form

    Returns:
        Token response with access token
    """
    user = authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=access_token_expires,
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
    )


@router.post("/validate")
async def validate_token(token: str) -> Dict[str, bool]:
    """
    Validate a token.

    Args:
        token: JWT token to validate

    Returns:
        Dictionary with validation result
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")

        if username is None:
            return {"valid": False}

        user = get_user(username)

        if user is None:
            return {"valid": False}

        return {"valid": True}
    except JWTError:
        return {"valid": False}