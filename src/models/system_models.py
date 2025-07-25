from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field

from .enums import QueryMode, ValidationType


# ============================================================================
# System Capabilities and Status Models
# ============================================================================

class SystemCapabilities(BaseModel):
    """System capabilities and status."""
    available_modes: List[QueryMode]
    validation_enabled: bool = True
    max_query_length: int = 1000
    supported_languages: List[str] = ["zh", "en"]
    version: str = "1.0.0"
    vector_store_status: str = "connected"
    redis_status: str = "connected"
    job_queue_status: str = "active"
    system_health: str = "healthy"
    component_health: Dict[str, str] = {}
    system_stats: Dict[str, Any] = {}
    features: Dict[str, Any] = {}


class SystemHealth(BaseModel):
    """System health check response."""
    status: str = "healthy"  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    components: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}
    uptime_seconds: float = 0.0
    version: str = "1.0.0"


class ComponentStatus(BaseModel):
    """Status of individual system components."""
    component_name: str
    status: str  # "healthy", "degraded", "offline", "error"
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


class SystemMetrics(BaseModel):
    """System performance metrics."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_io_mbps: float
    active_connections: int
    requests_per_second: float
    average_response_time_ms: float
    error_rate_percent: float


# ============================================================================
# Version and Feature Models
# ============================================================================

class VersionInfo(BaseModel):
    """System version information."""
    version: str
    build_date: datetime
    git_commit: Optional[str] = None
    branch: Optional[str] = None
    environment: str  # "development", "staging", "production"
    features_enabled: List[str] = []
    api_version: str = "v1"


class FeatureFlag(BaseModel):
    """Feature flag configuration."""
    flag_name: str
    enabled: bool
    description: str
    rollout_percentage: float = 100.0
    target_users: List[str] = []
    target_groups: List[str] = []
    conditions: Dict[str, Any] = {}


class SystemConfiguration(BaseModel):
    """System configuration settings."""
    config_version: str
    debug_mode: bool = False
    log_level: str = "INFO"
    max_concurrent_jobs: int = 100
    job_timeout_seconds: int = 1800
    database_settings: Dict[str, Any] = {}
    cache_settings: Dict[str, Any] = {}
    external_services: Dict[str, str] = {}


# ============================================================================
# Error and Debug Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    job_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DebugInfo(BaseModel):
    """Debug information for development."""
    query: str
    documents_retrieved: int
    search_time_ms: float
    processing_steps: List[str] = []
    internal_metrics: Dict[str, Any] = {}
    memory_usage: Optional[float] = None
    performance_stats: Dict[str, float] = {}


class SystemAlert(BaseModel):
    """System alert for monitoring."""
    alert_id: str
    alert_type: str  # "error", "warning", "info"
    severity: int  # 1-5, where 5 is critical
    component: str
    message: str
    details: Dict[str, Any] = {}
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


# ============================================================================
# Resource and Capacity Models
# ============================================================================

class ResourceUsage(BaseModel):
    """Current resource usage."""
    cpu_cores_used: float
    memory_mb_used: float
    disk_gb_used: float
    network_mbps_used: float
    database_connections: int
    cache_entries: int
    active_jobs: int


class CapacityLimits(BaseModel):
    """System capacity limits."""
    max_cpu_cores: float
    max_memory_mb: float
    max_disk_gb: float
    max_network_mbps: float
    max_database_connections: int
    max_cache_entries: int
    max_concurrent_jobs: int
    max_requests_per_second: float


class ResourceAllocation(BaseModel):
    """Resource allocation for different components."""
    component_name: str
    allocated_cpu_percent: float
    allocated_memory_mb: float
    allocated_disk_gb: float
    priority: int  # 1-10, higher is more priority
    scaling_enabled: bool = False
    min_instances: int = 1
    max_instances: int = 1


# ============================================================================
# Service Discovery and Registry Models
# ============================================================================

class ServiceInfo(BaseModel):
    """Information about a registered service."""
    service_id: str
    service_name: str
    version: str
    host: str
    port: int
    health_check_url: str
    status: str  # "active", "inactive", "unhealthy"
    last_heartbeat: datetime
    metadata: Dict[str, str] = {}


class ServiceRegistry(BaseModel):
    """Registry of available services."""
    registry_id: str
    services: List[ServiceInfo] = []
    last_updated: datetime
    registry_health: str = "healthy"


# ============================================================================
# Monitoring and Observability Models
# ============================================================================

class LogEntry(BaseModel):
    """Structured log entry."""
    timestamp: datetime
    level: str  # "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    logger: str
    message: str
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    job_id: Optional[str] = None
    extra_data: Dict[str, Any] = {}


class MetricPoint(BaseModel):
    """Individual metric data point."""
    metric_name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = {}
    unit: Optional[str] = None


class TraceSpan(BaseModel):
    """Distributed tracing span."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str  # "ok", "error", "timeout"
    tags: Dict[str, str] = {}
    logs: List[Dict[str, Any]] = []


# ============================================================================
# Backup and Recovery Models
# ============================================================================

class BackupInfo(BaseModel):
    """Information about system backups."""
    backup_id: str
    backup_type: str  # "full", "incremental", "differential"
    created_at: datetime
    size_bytes: int
    duration_seconds: float
    status: str  # "completed", "failed", "in_progress"
    storage_location: str
    verification_status: str  # "verified", "failed", "not_verified"


class RecoveryPlan(BaseModel):
    """Disaster recovery plan information."""
    plan_id: str
    plan_name: str
    recovery_time_objective: int  # seconds
    recovery_point_objective: int  # seconds
    backup_frequency: str
    last_test_date: Optional[datetime] = None
    test_success: Optional[bool] = None
    procedures: List[str] = []


# ============================================================================
# Cache and Storage Models
# ============================================================================

class CacheStats(BaseModel):
    """Cache statistics."""
    cache_name: str
    total_entries: int
    hit_rate_percent: float
    miss_rate_percent: float
    eviction_count: int
    memory_usage_mb: float
    max_memory_mb: float
    average_access_time_ms: float


class StorageStats(BaseModel):
    """Storage system statistics."""
    storage_type: str  # "database", "file_system", "object_storage"
    total_size_gb: float
    used_size_gb: float
    available_size_gb: float
    read_iops: float
    write_iops: float
    read_throughput_mbps: float
    write_throughput_mbps: float


# ============================================================================
# API and Interface Models
# ============================================================================

class APIEndpoint(BaseModel):
    """API endpoint information."""
    path: str
    method: str
    description: str
    version: str
    deprecated: bool = False
    rate_limit: Optional[int] = None
    authentication_required: bool = True
    permissions_required: List[str] = []


class APIUsageStats(BaseModel):
    """API usage statistics."""
    endpoint: str
    method: str
    total_requests: int
    successful_requests: int
    error_requests: int
    average_response_time_ms: float
    p95_response_time_ms: float
    rate_limit_hits: int
    last_24h_requests: int


class IntegrationStatus(BaseModel):
    """Status of external integrations."""
    integration_name: str
    integration_type: str  # "api", "webhook", "database", "message_queue"
    status: str  # "connected", "disconnected", "error", "rate_limited"
    last_successful_call: Optional[datetime] = None
    error_count: int = 0
    last_error: Optional[str] = None
    configuration: Dict[str, Any] = {}


# ============================================================================
# Maintenance and Operations Models
# ============================================================================

class MaintenanceWindow(BaseModel):
    """Scheduled maintenance window."""
    window_id: str
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    maintenance_type: str  # "scheduled", "emergency", "security"
    affected_services: List[str] = []
    impact_level: str  # "low", "medium", "high", "critical"
    status: str  # "scheduled", "in_progress", "completed", "cancelled"


class DeploymentInfo(BaseModel):
    """Information about system deployments."""
    deployment_id: str
    version: str
    environment: str
    deployed_at: datetime
    deployed_by: str
    rollback_available: bool = True
    health_check_passed: bool = True
    features_changed: List[str] = []
    breaking_changes: List[str] = []


class SystemEvent(BaseModel):
    """System-level event."""
    event_id: str
    event_type: str  # "startup", "shutdown", "deployment", "error", "maintenance"
    timestamp: datetime
    severity: str  # "info", "warning", "error", "critical"
    message: str
    details: Dict[str, Any] = {}
    affected_components: List[str] = []
    resolution_status: str = "open"  # "open", "investigating", "resolved"


# ============================================================================
# Configuration and Settings Models
# ============================================================================

class ConfigurationSetting(BaseModel):
    """Individual configuration setting."""
    key: str
    value: Any
    data_type: str  # "string", "integer", "float", "boolean", "json"
    description: str
    default_value: Any
    requires_restart: bool = False
    environment_specific: bool = False
    last_modified: datetime
    modified_by: Optional[str] = None


class EnvironmentConfig(BaseModel):
    """Environment-specific configuration."""
    environment_name: str
    settings: List[ConfigurationSetting] = []
    secrets: List[str] = []  # Just keys, not values
    feature_flags: List[FeatureFlag] = []
    last_updated: datetime


# ============================================================================
# Utility and Helper Models
# ============================================================================

class ValidationMetrics(BaseModel):
    """Metrics for validation performance tracking."""
    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    user_interventions: int = 0
    average_confidence: float = 0.0
    average_processing_time: float = 0.0
    user_satisfaction_rate: float = 0.0


class SystemHealthStatus(BaseModel):
    """Overall system health including validation capabilities."""
    status: Literal["healthy", "degraded", "maintenance"]
    validation_system_status: Literal["operational", "limited", "offline"]
    active_validations: int = 0
    queue_status: Dict[str, Any] = {}
    last_updated: datetime = Field(default_factory=datetime.now)


class Notification(BaseModel):
    """System notification model."""
    notification_id: str
    notification_type: str  # "info", "warning", "error", "maintenance"
    title: str
    message: str
    target_users: List[str] = []
    target_roles: List[str] = []
    created_at: datetime
    expires_at: Optional[datetime] = None
    read_by: List[str] = []
    dismissed_by: List[str] = []