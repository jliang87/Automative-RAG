from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field

from .enums import SourceType, ValidationStepType


# ============================================================================
# Automotive Reference Data Models
# ============================================================================

class AutomotiveSpecification(BaseModel):
    """Structured automotive specification data"""
    spec_id: str
    spec_type: str  # "fuel_economy", "engine", "safety", "dimensions", "pricing"

    # Vehicle identification
    manufacturer: str
    model: str
    year: int
    trim: Optional[str] = None
    market: str = "US"

    # Specification data
    specifications: Dict[str, Any]  # The actual spec values
    tolerance_ranges: Dict[str, float] = {}  # Acceptable variance

    # Source and quality
    source_type: SourceType
    source_url: Optional[str] = None
    authority_score: float = Field(ge=0.0, le=1.0)
    verified: bool = False
    last_verified: Optional[datetime] = None

    # Usage tracking
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class FuelEconomyReference(BaseModel):
    """EPA and manufacturer fuel economy data"""
    reference_id: str
    manufacturer: str
    model: str
    year: int
    trim: Optional[str] = None
    engine_type: Optional[str] = None

    # Fuel economy data
    city_mpg: Optional[int] = None
    highway_mpg: Optional[int] = None
    combined_mpg: Optional[int] = None
    fuel_type: str = "Regular Gasoline"

    # EPA specific
    epa_vehicle_id: Optional[str] = None

    # Source information
    source_type: SourceType
    source_url: Optional[str] = None
    verified_date: Optional[datetime] = None


class EngineSpecification(BaseModel):
    """Engine specification reference data"""
    spec_id: str
    manufacturer: str
    model: str
    year: int
    engine_code: Optional[str] = None

    # Engine specifications
    horsepower: Optional[int] = None
    torque: Optional[int] = None  # lb-ft
    displacement: Optional[float] = None  # liters
    engine_type: Optional[str] = None  # "V6", "I4", "V8", etc.
    fuel_type: Optional[str] = None

    # Performance data
    acceleration_0_60: Optional[float] = None  # seconds
    top_speed: Optional[int] = None  # mph

    # Source information
    source_type: SourceType
    verified: bool = False


class SafetyRating(BaseModel):
    """Safety rating reference data"""
    rating_id: str
    manufacturer: str
    model: str
    year: int

    # NHTSA ratings
    nhtsa_overall: Optional[int] = None  # 1-5 stars
    nhtsa_frontal: Optional[int] = None
    nhtsa_side: Optional[int] = None
    nhtsa_rollover: Optional[int] = None

    # IIHS ratings
    iihs_top_safety_pick: bool = False
    iihs_top_safety_pick_plus: bool = False
    iihs_awards: List[str] = []

    # Source information
    source_type: SourceType = SourceType.REGULATORY
    test_date: Optional[datetime] = None


class VehicleDimensions(BaseModel):
    """Vehicle dimension reference data"""
    spec_id: str
    manufacturer: str
    model: str
    year: int

    # Dimensions (inches)
    length: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    wheelbase: Optional[float] = None

    # Weight (pounds)
    curb_weight: Optional[int] = None
    gross_weight: Optional[int] = None

    # Cargo
    cargo_volume: Optional[float] = None  # cubic feet
    passenger_volume: Optional[float] = None

    # Source information
    source_type: SourceType
    verified: bool = False


# ============================================================================
# Source Authority Models
# ============================================================================

class SourceAuthorityDatabase(BaseModel):
    """Complete source authority database"""
    authorities: List['SourceAuthority'] = []
    last_updated: datetime = Field(default_factory=datetime.now)
    version: str = "1.0"


class SourceAuthority(BaseModel):
    """Source authority and credibility scoring"""
    domain: str
    source_type: SourceType

    # Authority metrics
    authority_score: float = Field(ge=0.0, le=1.0)
    bias_score: float = Field(ge=0.0, le=1.0, default=0.0)
    reliability_score: float = Field(ge=0.0, le=1.0, default=1.0)

    # Expertise information
    expertise_level: str = "general"  # "expert", "professional", "enthusiast", "general"
    specializations: List[str] = []  # ["bmw", "electric_vehicles", "safety"]
    coverage_areas: List[str] = []  # ["specifications", "reviews", "ratings"]

    # Quality factors
    accuracy_history: float = Field(ge=0.0, le=1.0, default=1.0)
    peer_ratings: List[float] = []
    verification_count: int = 0

    # Metadata
    description: Optional[str] = None
    last_verified: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)


# ============================================================================
# Physics and Validation Constraint Models
# ============================================================================

class PhysicsConstraint(BaseModel):
    """Physics-based validation constraints"""
    constraint_id: str
    constraint_type: str  # "fuel_economy", "engine", "performance", "dimensions"

    # Constraint definition
    rule_expression: str  # "highway_mpg >= city_mpg"
    description: str
    severity: str = "error"  # "error", "warning", "info"

    # Applicable scope
    applies_to: List[str] = []  # ["all", "gasoline", "hybrid", "electric"]
    exceptions: List[str] = []

    # Ranges
    reasonable_ranges: Dict[str, Dict[str, float]] = {}  # {"city_mpg": {"min": 8, "max": 60}}


class ValidationConstraints(BaseModel):
    """Complete set of validation constraints"""
    fuel_economy_constraints: List[PhysicsConstraint] = []
    engine_constraints: List[PhysicsConstraint] = []
    dimension_constraints: List[PhysicsConstraint] = []
    performance_constraints: List[PhysicsConstraint] = []

    # Global settings
    default_tolerance: float = 0.1  # 10% default tolerance
    strict_mode: bool = False


# ============================================================================
# Validation Reference Management Models
# ============================================================================

class ValidationReferenceDatabase(BaseModel):
    """Complete validation reference database"""
    fuel_economy_data: List[FuelEconomyReference] = []
    engine_specifications: List[EngineSpecification] = []
    safety_ratings: List[SafetyRating] = []
    vehicle_dimensions: List[VehicleDimensions] = []

    # Metadata
    last_updated: datetime = Field(default_factory=datetime.now)
    total_entries: int = 0
    coverage_stats: Dict[str, int] = {}


class ReferenceDataQuery(BaseModel):
    """Query interface for reference data"""
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    year_range: Optional[tuple[int, int]] = None
    spec_types: Optional[List[str]] = None
    source_types: Optional[List[SourceType]] = None
    verified_only: bool = False

    # Results control
    limit: int = 100
    offset: int = 0


class ReferenceDataMatch(BaseModel):
    """Result of reference data matching"""
    match_type: str  # "exact", "partial", "fuzzy", "none"
    confidence: float = Field(ge=0.0, le=1.0)
    matched_reference: Optional[
        Union[FuelEconomyReference, EngineSpecification, SafetyRating, VehicleDimensions]] = None
    match_criteria: Dict[str, str] = {}
    alternatives: List[Any] = []


# ============================================================================
# Consensus Analysis Models
# ============================================================================

class FactualClaim(BaseModel):
    """Structured factual claim for consensus analysis"""
    claim_id: str
    claim_type: str  # "numerical", "rating", "comparative", "absolute"
    claim_category: str  # "fuel_economy", "performance", "safety", etc.

    # Claim data
    claim_text: str
    extracted_value: Any
    context: str
    confidence: float = Field(ge=0.0, le=1.0)

    # Source information
    source_id: str
    source_type: SourceType
    source_authority: float
    document_index: int


class ConsensusAnalysisResult(BaseModel):
    """Result of consensus analysis"""
    category: str
    total_claims: int

    # Consensus metrics
    consensus_strength: float = Field(ge=0.0, le=1.0)
    agreement_count: int
    disagreement_count: int

    # Claim analysis
    claims: List[FactualClaim] = []
    consensus_value: Optional[Any] = None
    disagreements: List[Dict[str, Any]] = []

    # Source analysis
    source_diversity: float = Field(ge=0.0, le=1.0)
    authority_weighted_consensus: float = Field(ge=0.0, le=1.0)


# ============================================================================
# Confidence Calculation Models
# ============================================================================

class ConfidenceWeights(BaseModel):
    """Weights for confidence calculation"""
    source_credibility: float = 0.4
    technical_consistency: float = 0.3
    completeness: float = 0.2
    consensus: float = 0.1
    llm_inference: float = 0.1
    retrieval: float = 0.05


class ConfidenceCalculationConfig(BaseModel):
    """Configuration for confidence calculation"""
    weights: ConfidenceWeights = ConfidenceWeights()

    # Thresholds
    high_confidence_threshold: float = 0.8
    medium_confidence_threshold: float = 0.6

    # Penalties and bonuses
    unverifiable_penalty: float = 0.3
    physics_violation_penalty: float = 0.2
    authority_bonus_threshold: float = 0.9

    # Meta-validation settings
    enable_coverage_penalty: bool = True
    enable_quality_multiplier: bool = True


# ============================================================================
# Auto-Fetch Models
# ============================================================================

class AutoFetchTarget(BaseModel):
    """Target for auto-fetch operations"""
    target_type: str  # "epa_data", "manufacturer_specs", "safety_ratings"
    target_url: str
    expected_data_type: str
    confidence_boost_estimate: float

    # Query context
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None


class AutoFetchResult(BaseModel):
    """Result from auto-fetch operation"""
    target_type: str
    success: bool

    # Fetched data
    documents_added: int = 0
    data_fetched: List[Dict[str, Any]] = []

    # Quality assessment
    authority_score: float = 0.0
    data_quality: str = "unknown"  # "high", "medium", "low", "unknown"

    # Processing info
    fetch_duration: float = 0.0
    error_details: Optional[str] = None


# ============================================================================
# Data Migration Models
# ============================================================================

class HardcodedDataMigration(BaseModel):
    """Migration record for hardcoded data"""
    migration_id: str
    source_file: str
    source_function: str
    migration_type: str  # "authority_scores", "reference_data", "constraints"

    # Migration data
    original_data: Dict[str, Any]
    migrated_entries: List[str]  # IDs of created model instances

    # Status
    status: str = "pending"  # "pending", "completed", "failed"
    migration_date: Optional[datetime] = None
    error_details: Optional[str] = None


# ============================================================================
# Helper Functions for Model Creation
# ============================================================================

def create_epa_fuel_economy_reference(vehicle_data: Dict[str, Any]) -> FuelEconomyReference:
    """Helper to create EPA fuel economy reference from data"""
    return FuelEconomyReference(
        reference_id=f"epa_{vehicle_data['year']}_{vehicle_data['manufacturer']}_{vehicle_data['model']}",
        manufacturer=vehicle_data["manufacturer"],
        model=vehicle_data["model"],
        year=vehicle_data["year"],
        city_mpg=vehicle_data.get("city_mpg"),
        highway_mpg=vehicle_data.get("highway_mpg"),
        combined_mpg=vehicle_data.get("combined_mpg"),
        source_type=SourceType.REGULATORY,
        source_url="https://www.fueleconomy.gov",
        verified_date=datetime.now()
    )


def create_source_authority(domain: str, authority_data: Dict[str, Any]) -> SourceAuthority:
    """Helper to create source authority from hardcoded data"""
    return SourceAuthority(
        domain=domain,
        source_type=authority_data.get("type", SourceType.PROFESSIONAL),
        authority_score=authority_data.get("authority", 0.5),
        bias_score=authority_data.get("bias", 0.0),
        description=f"Authority data for {domain}",
        created_at=datetime.now()
    )


def create_physics_constraint(constraint_data: Dict[str, Any]) -> PhysicsConstraint:
    """Helper to create physics constraint from hardcoded rules"""
    return PhysicsConstraint(
        constraint_id=f"physics_{constraint_data['type']}_{len(constraint_data.get('rules', []))}",
        constraint_type=constraint_data["type"],
        rule_expression=constraint_data["rule"],
        description=constraint_data.get("description", "Physics constraint"),
        reasonable_ranges=constraint_data.get("ranges", {})
    )