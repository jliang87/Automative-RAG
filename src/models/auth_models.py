from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ============================================================================
# Authentication Models
# ============================================================================

class TokenResponse(BaseModel):
    """Response containing authentication token."""
    access_token: str
    token_type: str


class TokenRequest(BaseModel):
    """Request for authentication token."""
    username: str
    password: str


class UserInfo(BaseModel):
    """User information model."""
    user_id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: bool = True
    roles: List[str] = []
    permissions: List[str] = []
    created_at: datetime
    last_login: Optional[datetime] = None


class UserSession(BaseModel):
    """User session information."""
    session_id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True


class Permission(BaseModel):
    """Permission model."""
    permission_id: str
    name: str
    description: str
    resource: str
    action: str  # "create", "read", "update", "delete", "execute"


class Role(BaseModel):
    """Role model."""
    role_id: str
    name: str
    description: str
    permissions: List[str] = []  # Permission IDs
    is_default: bool = False


# ============================================================================
# API Key Models
# ============================================================================

class APIKey(BaseModel):
    """API key model."""
    key_id: str
    key_hash: str
    name: str
    user_id: str
    scopes: List[str] = []
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True


class APIKeyRequest(BaseModel):
    """Request to create API key."""
    name: str
    scopes: List[str] = []
    expires_in_days: Optional[int] = None


class APIKeyResponse(BaseModel):
    """Response containing new API key."""
    key_id: str
    api_key: str  # Only returned once
    name: str
    scopes: List[str]
    expires_at: Optional[datetime] = None


# ============================================================================
# OAuth and External Auth Models
# ============================================================================

class OAuthProvider(BaseModel):
    """OAuth provider configuration."""
    provider_id: str
    name: str
    client_id: str
    authorization_url: str
    token_url: str
    user_info_url: str
    scopes: List[str] = []
    is_enabled: bool = True


class OAuthToken(BaseModel):
    """OAuth token information."""
    token_id: str
    user_id: str
    provider_id: str
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: datetime
    scopes: List[str] = []


class ExternalUserInfo(BaseModel):
    """User information from external OAuth provider."""
    external_id: str
    provider_id: str
    email: str
    username: Optional[str] = None
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    verified_email: bool = False


# ============================================================================
# Security and Audit Models
# ============================================================================

class LoginAttempt(BaseModel):
    """Login attempt record."""
    attempt_id: str
    username: str
    ip_address: str
    user_agent: str
    success: bool
    failure_reason: Optional[str] = None
    timestamp: datetime


class SecurityEvent(BaseModel):
    """Security-related event."""
    event_id: str
    event_type: str  # "login", "logout", "password_change", "permission_change"
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    details: Dict[str, Any] = {}
    timestamp: datetime
    severity: str = "info"  # "info", "warning", "critical"


class AuditLog(BaseModel):
    """Audit log entry."""
    log_id: str
    user_id: Optional[str] = None
    action: str
    resource: str
    resource_id: Optional[str] = None
    old_values: Optional[Dict[str, Any]] = None
    new_values: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime


# ============================================================================
# Password and Security Models
# ============================================================================

class PasswordPolicy(BaseModel):
    """Password policy configuration."""
    min_length: int = 8
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_symbols: bool = False
    max_age_days: Optional[int] = None
    prevent_reuse_count: int = 5


class PasswordReset(BaseModel):
    """Password reset request."""
    reset_id: str
    user_id: str
    token_hash: str
    created_at: datetime
    expires_at: datetime
    used: bool = False
    ip_address: Optional[str] = None


class PasswordChangeRequest(BaseModel):
    """Request to change password."""
    current_password: str
    new_password: str
    confirm_password: str


class PasswordResetRequest(BaseModel):
    """Request to reset password."""
    email: str
    username: Optional[str] = None


# ============================================================================
# Multi-Factor Authentication Models
# ============================================================================

class MFADevice(BaseModel):
    """Multi-factor authentication device."""
    device_id: str
    user_id: str
    device_type: str  # "totp", "sms", "email", "hardware"
    device_name: str
    secret_key: Optional[str] = None  # For TOTP
    phone_number: Optional[str] = None  # For SMS
    is_verified: bool = False
    is_primary: bool = False
    created_at: datetime
    last_used: Optional[datetime] = None


class MFAChallenge(BaseModel):
    """MFA challenge for authentication."""
    challenge_id: str
    user_id: str
    device_id: str
    challenge_type: str
    created_at: datetime
    expires_at: datetime
    attempts: int = 0
    max_attempts: int = 3
    verified: bool = False


class MFAVerification(BaseModel):
    """MFA verification request."""
    challenge_id: str
    verification_code: str


# ============================================================================
# Rate Limiting and Security Models
# ============================================================================

class RateLimit(BaseModel):
    """Rate limiting configuration."""
    resource: str
    limit_type: str  # "requests_per_minute", "requests_per_hour", "requests_per_day"
    limit_value: int
    window_size: int  # in seconds
    burst_limit: Optional[int] = None


class RateLimitStatus(BaseModel):
    """Current rate limit status for a user/resource."""
    resource: str
    user_id: Optional[str] = None
    requests_made: int
    limit_value: int
    window_start: datetime
    window_end: datetime
    blocked: bool = False


class SecuritySettings(BaseModel):
    """Security settings for the system."""
    session_timeout: int = 3600  # seconds
    max_concurrent_sessions: int = 5
    password_policy: PasswordPolicy
    mfa_required: bool = False
    rate_limits: List[RateLimit] = []
    allowed_origins: List[str] = []
    blocked_ips: List[str] = []


# ============================================================================
# Error Models
# ============================================================================

class AuthenticationError(BaseModel):
    """Authentication error response."""
    error: str
    error_description: str
    error_code: Optional[str] = None
    timestamp: datetime


class AuthorizationError(BaseModel):
    """Authorization error response."""
    error: str
    required_permission: str
    user_permissions: List[str]
    resource: str
    action: str
    timestamp: datetime