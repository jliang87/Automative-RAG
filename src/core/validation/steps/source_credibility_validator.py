"""
Source Credibility Validator - COMPLETE UPDATED VERSION
Validates source authority and credibility using model-driven authority database
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.models import (
    ValidationStepResult, ValidationStatus, ValidationStepType,
    ValidationContext, ValidationWarning, SourceType
)
from src.models.knowledge_models import SourceAuthority, SourceAuthorityDatabase
from .steps_readiness_checker import MetaValidator, PreconditionResult

logger = logging.getLogger(__name__)


class SourceCredibilityValidator:
    """
    Validates the credibility and authority of information sources using model-driven approach
    """

    def __init__(self,
                 step_config: Dict[str, Any],
                 meta_validator: MetaValidator,
                 authority_database: Optional[SourceAuthorityDatabase] = None):
        self.step_config = step_config
        self.meta_validator = meta_validator
        self.step_type = ValidationStepType.SOURCE_CREDIBILITY

        # Use model-based authority database instead of hardcoded scores
        self.authority_database = authority_database or self._load_default_authority_database()

        # Build lookup for performance
        self.authority_lookup = {
            auth.domain: auth for auth in self.authority_database.authorities
        }

    def _load_default_authority_database(self) -> SourceAuthorityDatabase:
        """Load default authority database using SourceAuthority models instead of hardcoded dict"""

        # Create model instances instead of hardcoded dict
        authorities = [
            # Official Sources (High Authority)
            SourceAuthority(
                domain="epa.gov",
                source_type=SourceType.REGULATORY,
                authority_score=1.0,
                bias_score=0.0,
                reliability_score=1.0,
                expertise_level="expert",
                specializations=["fuel_economy", "emissions", "environmental_standards"],
                coverage_areas=["specifications", "testing", "regulations"],
                accuracy_history=1.0,
                description="US Environmental Protection Agency - Official fuel economy and emissions data",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="nhtsa.gov",
                source_type=SourceType.REGULATORY,
                authority_score=1.0,
                bias_score=0.0,
                reliability_score=1.0,
                expertise_level="expert",
                specializations=["safety", "crash_testing", "vehicle_standards"],
                coverage_areas=["safety_ratings", "recalls", "standards"],
                accuracy_history=1.0,
                description="National Highway Traffic Safety Administration - Official safety ratings",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="toyota.com",
                source_type=SourceType.OFFICIAL,
                authority_score=0.95,
                bias_score=0.1,
                reliability_score=0.95,
                expertise_level="expert",
                specializations=["toyota", "lexus", "manufacturing"],
                coverage_areas=["specifications", "features", "pricing"],
                accuracy_history=0.95,
                description="Toyota Motor Corporation Official Website",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="honda.com",
                source_type=SourceType.OFFICIAL,
                authority_score=0.95,
                bias_score=0.1,
                reliability_score=0.95,
                expertise_level="expert",
                specializations=["honda", "acura", "manufacturing"],
                coverage_areas=["specifications", "features", "pricing"],
                accuracy_history=0.95,
                description="Honda Motor Company Official Website",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="ford.com",
                source_type=SourceType.OFFICIAL,
                authority_score=0.95,
                bias_score=0.1,
                reliability_score=0.95,
                expertise_level="expert",
                specializations=["ford", "lincoln", "manufacturing"],
                coverage_areas=["specifications", "features", "pricing"],
                accuracy_history=0.95,
                description="Ford Motor Company Official Website",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="bmw.com",
                source_type=SourceType.OFFICIAL,
                authority_score=0.95,
                bias_score=0.1,
                reliability_score=0.95,
                expertise_level="expert",
                specializations=["bmw", "mini", "luxury_vehicles"],
                coverage_areas=["specifications", "features", "pricing"],
                accuracy_history=0.95,
                description="BMW Group Official Website",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="mercedes-benz.com",
                source_type=SourceType.OFFICIAL,
                authority_score=0.95,
                bias_score=0.1,
                reliability_score=0.95,
                expertise_level="expert",
                specializations=["mercedes", "luxury_vehicles", "manufacturing"],
                coverage_areas=["specifications", "features", "pricing"],
                accuracy_history=0.95,
                description="Mercedes-Benz Official Website",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),

            # Professional Automotive Media (Medium-High Authority)
            SourceAuthority(
                domain="caranddriver.com",
                source_type=SourceType.PROFESSIONAL,
                authority_score=0.85,
                bias_score=0.2,
                reliability_score=0.85,
                expertise_level="professional",
                specializations=["reviews", "testing", "automotive_journalism"],
                coverage_areas=["reviews", "comparisons", "testing"],
                accuracy_history=0.85,
                peer_ratings=[0.9, 0.8, 0.85, 0.9],
                description="Car and Driver Magazine - Professional automotive reviews and testing",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="motortrend.com",
                source_type=SourceType.PROFESSIONAL,
                authority_score=0.85,
                bias_score=0.2,
                reliability_score=0.85,
                expertise_level="professional",
                specializations=["reviews", "testing", "awards", "automotive_journalism"],
                coverage_areas=["reviews", "comparisons", "awards"],
                accuracy_history=0.85,
                peer_ratings=[0.85, 0.9, 0.8, 0.85],
                description="MotorTrend Magazine - Professional automotive reviews and awards",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="roadandtrack.com",
                source_type=SourceType.PROFESSIONAL,
                authority_score=0.80,
                bias_score=0.2,
                reliability_score=0.80,
                expertise_level="professional",
                specializations=["performance", "racing", "enthusiast_content"],
                coverage_areas=["reviews", "performance", "racing"],
                accuracy_history=0.80,
                description="Road & Track Magazine - Performance and enthusiast content",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="automobilemag.com",
                source_type=SourceType.PROFESSIONAL,
                authority_score=0.80,
                bias_score=0.2,
                reliability_score=0.80,
                expertise_level="professional",
                specializations=["reviews", "features", "automotive_journalism"],
                coverage_areas=["reviews", "features", "news"],
                accuracy_history=0.80,
                description="Automobile Magazine - Professional automotive journalism",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="autoweek.com",
                source_type=SourceType.PROFESSIONAL,
                authority_score=0.75,
                bias_score=0.2,
                reliability_score=0.75,
                expertise_level="professional",
                specializations=["news", "reviews", "industry_analysis"],
                coverage_areas=["news", "reviews", "industry"],
                accuracy_history=0.75,
                description="Autoweek Magazine - Automotive news and reviews",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="kbb.com",
                source_type=SourceType.PROFESSIONAL,
                authority_score=0.85,
                bias_score=0.15,
                reliability_score=0.85,
                expertise_level="professional",
                specializations=["pricing", "valuation", "market_analysis"],
                coverage_areas=["pricing", "reviews", "market_data"],
                accuracy_history=0.90,
                description="Kelley Blue Book - Vehicle pricing and valuation authority",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="edmunds.com",
                source_type=SourceType.PROFESSIONAL,
                authority_score=0.85,
                bias_score=0.15,
                reliability_score=0.85,
                expertise_level="professional",
                specializations=["pricing", "reviews", "market_analysis", "specifications"],
                coverage_areas=["pricing", "reviews", "specifications"],
                accuracy_history=0.85,
                description="Edmunds - Comprehensive automotive information and pricing",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),

            # Consumer/Review Sites (Medium Authority)
            SourceAuthority(
                domain="consumerreports.org",
                source_type=SourceType.PROFESSIONAL,
                authority_score=0.90,
                bias_score=0.1,
                reliability_score=0.95,
                expertise_level="professional",
                specializations=["reliability", "testing", "consumer_advocacy"],
                coverage_areas=["reliability", "testing", "ratings"],
                accuracy_history=0.95,
                description="Consumer Reports - Independent testing and reliability data",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="carsguide.com.au",
                source_type=SourceType.PROFESSIONAL,
                authority_score=0.70,
                bias_score=0.3,
                reliability_score=0.70,
                expertise_level="professional",
                specializations=["reviews", "australian_market"],
                coverage_areas=["reviews", "pricing", "market_data"],
                accuracy_history=0.70,
                description="CarsGuide Australia - Australian automotive marketplace",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="cars.com",
                source_type=SourceType.PROFESSIONAL,
                authority_score=0.70,
                bias_score=0.25,
                reliability_score=0.70,
                expertise_level="professional",
                specializations=["marketplace", "reviews", "pricing"],
                coverage_areas=["marketplace", "reviews", "pricing"],
                accuracy_history=0.70,
                description="Cars.com - Automotive marketplace and reviews",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),

            # User-Generated Content (Lower Authority)
            SourceAuthority(
                domain="reddit.com",
                source_type=SourceType.USER_GENERATED,
                authority_score=0.40,
                bias_score=0.4,
                reliability_score=0.40,
                expertise_level="enthusiast",
                specializations=["discussions", "experiences", "community"],
                coverage_areas=["discussions", "experiences", "opinions"],
                accuracy_history=0.50,
                description="Reddit - Community discussions and user experiences",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="forums.edmunds.com",
                source_type=SourceType.USER_GENERATED,
                authority_score=0.45,
                bias_score=0.4,
                reliability_score=0.45,
                expertise_level="enthusiast",
                specializations=["discussions", "experiences", "technical"],
                coverage_areas=["discussions", "experiences", "technical"],
                accuracy_history=0.50,
                description="Edmunds Forums - User discussions and experiences",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="carsguru.net",
                source_type=SourceType.USER_GENERATED,
                authority_score=0.35,
                bias_score=0.5,
                reliability_score=0.35,
                expertise_level="general",
                specializations=["discussions", "reviews"],
                coverage_areas=["discussions", "user_reviews"],
                accuracy_history=0.40,
                description="CarsGuru - User-generated automotive content",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),

            # Academic/Research (High Authority)
            SourceAuthority(
                domain="sae.org",
                source_type=SourceType.ACADEMIC,
                authority_score=0.95,
                bias_score=0.05,
                reliability_score=0.95,
                expertise_level="expert",
                specializations=["engineering", "research", "standards", "technical"],
                coverage_areas=["research", "standards", "technical"],
                accuracy_history=0.95,
                description="Society of Automotive Engineers - Technical standards and research",
                last_verified=datetime.now(),
                created_at=datetime.now()
            ),
            SourceAuthority(
                domain="ieee.org",
                source_type=SourceType.ACADEMIC,
                authority_score=0.95,
                bias_score=0.05,
                reliability_score=0.95,
                expertise_level="expert",
                specializations=["engineering", "research", "standards", "electrical"],
                coverage_areas=["research", "standards", "technical"],
                accuracy_history=0.95,
                description="Institute of Electrical and Electronics Engineers - Technical research",
                last_verified=datetime.now(),
                created_at=datetime.now()
            )
        ]

        return SourceAuthorityDatabase(
            authorities=authorities,
            last_updated=datetime.now(),
            version="1.0"
        )

    async def execute(self, context: ValidationContext) -> ValidationStepResult:
        """Execute source credibility validation"""

        start_time = datetime.now()

        # Check preconditions
        precondition_result = await self.meta_validator.check_preconditions(
            self.step_type, context, self.step_config
        )

        if precondition_result.status != "READY":
            return self._create_unverifiable_result(start_time, precondition_result)

        try:
            # Perform source credibility validation
            result = await self._perform_validation(context)
            result.completed_at = datetime.now()
            result.duration_ms = int((result.completed_at - start_time).total_seconds() * 1000)

            return result

        except Exception as e:
            logger.error(f"Source credibility validation failed: {str(e)}")
            return self._create_error_result(start_time, str(e))

    async def _perform_validation(self, context: ValidationContext) -> ValidationStepResult:
        """Perform the actual source credibility validation using model-driven approach"""

        documents = context.documents
        warnings = []
        sources_analyzed = []

        # Analyze each document's source credibility using model data
        source_scores = []
        source_details = {}

        for i, doc in enumerate(documents):
            source_analysis = self._analyze_document_source(doc, i)
            source_scores.append(source_analysis["authority_score"])
            sources_analyzed.append(source_analysis["source_identifier"])
            source_details[source_analysis["source_identifier"]] = source_analysis

            # Generate warnings for low-credibility sources
            if source_analysis["authority_score"] < 0.5:
                warnings.append(ValidationWarning(
                    category="low_authority",
                    severity="caution",
                    message=f"Low authority source: {source_analysis['source_identifier']}",
                    explanation=f"Authority score: {source_analysis['authority_score']:.2f}/1.0",
                    suggestion="Consider finding additional authoritative sources"
                ))

            # Generate warnings for high bias
            if source_analysis["bias_score"] > 0.4:
                warnings.append(ValidationWarning(
                    category="high_bias",
                    severity="caution",
                    message=f"Potential bias detected: {source_analysis['source_identifier']}",
                    explanation=f"Bias score: {source_analysis['bias_score']:.2f}/1.0",
                    suggestion="Cross-reference with more neutral sources"
                ))

        # Calculate overall source credibility metrics using model data
        avg_authority = sum(source_scores) / len(source_scores) if source_scores else 0.0
        source_diversity = len(set(source_details[src]["source_type"] for src in source_details))
        official_source_count = sum(1 for src in source_details.values()
                                    if src["source_type"] == SourceType.OFFICIAL)
        regulatory_source_count = sum(1 for src in source_details.values()
                                      if src["source_type"] == SourceType.REGULATORY)

        # Enhanced metrics using model data
        known_sources = sum(1 for src in source_details.values() if src.get("from_authority_database", False))
        avg_reliability = sum(
            source_details[src].get("reliability_score", 0.5) for src in source_details
        ) / len(source_details) if source_details else 0.0

        # Determine validation status using enhanced model-based criteria
        if avg_authority >= 0.8 and source_diversity >= 2 and regulatory_source_count > 0:
            status = ValidationStatus.PASSED
            confidence_impact = 18.0 + (avg_authority - 0.8) * 25  # 18-23 point boost
        elif avg_authority >= 0.8 and source_diversity >= 2:
            status = ValidationStatus.PASSED
            confidence_impact = 15.0 + (avg_authority - 0.8) * 25  # 15-20 point boost
        elif avg_authority >= 0.6:
            status = ValidationStatus.WARNING
            confidence_impact = 5.0 + (avg_authority - 0.6) * 25  # 5-10 point boost
        else:
            status = ValidationStatus.WARNING
            confidence_impact = max(-5.0, (avg_authority - 0.4) * 25)  # Small negative impact

            warnings.append(ValidationWarning(
                category="overall_credibility",
                severity="critical",
                message="Overall source credibility is low",
                explanation=f"Average authority score: {avg_authority:.2f}/1.0",
                suggestion="Add more authoritative sources like manufacturer websites or regulatory data"
            ))

        # Build summary with enhanced model information
        summary = (f"Analyzed {len(documents)} sources using model-driven authority database. "
                   f"Average authority: {avg_authority:.2f}/1.0, "
                   f"Source diversity: {source_diversity} types, "
                   f"Official sources: {official_source_count}, "
                   f"Regulatory sources: {regulatory_source_count}")

        # Build detailed results with model-based insights
        details = {
            "sources_analyzed": len(documents),
            "avg_authority_score": avg_authority,
            "avg_reliability_score": avg_reliability,
            "source_type_diversity": source_diversity,
            "official_sources_used": official_source_count,
            "regulatory_sources_used": regulatory_source_count,
            "known_sources_count": known_sources,
            "unknown_sources_count": len(documents) - known_sources,
            "source_breakdown": source_details,
            "authority_distribution": {
                "excellent": sum(1 for s in source_scores if s >= 0.9),
                "high": sum(1 for s in source_scores if 0.8 <= s < 0.9),
                "medium": sum(1 for s in source_scores if 0.5 <= s < 0.8),
                "low": sum(1 for s in source_scores if s < 0.5)
            },
            "bias_detected": any(src["bias_score"] > 0.4 for src in source_details.values()),
            "authority_database_version": self.authority_database.version,
            "authority_database_last_updated": self.authority_database.last_updated.isoformat(),
            "specialization_coverage": self._analyze_specialization_coverage(source_details),
            "source_type_distribution": self._get_source_type_distribution(source_details)
        }

        return ValidationStepResult(
            step_id=f"source_credibility_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Source Credibility Analysis",
            status=status,
            confidence_impact=confidence_impact,
            summary=summary,
            details=details,
            started_at=start_time,
            warnings=warnings,
            sources_used=sources_analyzed
        )

    def _analyze_document_source(self, doc: Dict[str, Any], doc_index: int) -> Dict[str, Any]:
        """Analyze the credibility of a single document's source using model data"""

        metadata = doc.get("metadata", {})

        # Extract source information
        source_url = metadata.get("url", "")
        source_domain = self._extract_domain(source_url)
        source_platform = metadata.get("sourcePlatform", metadata.get("source", "unknown"))

        # Look up in model-based authority database
        if source_domain in self.authority_lookup:
            authority_obj = self.authority_lookup[source_domain]
            authority_score = authority_obj.authority_score
            bias_score = authority_obj.bias_score
            source_type = authority_obj.source_type
            reliability_score = authority_obj.reliability_score
            expertise_level = authority_obj.expertise_level
            specializations = authority_obj.specializations
            from_authority_database = True
        else:
            # Assess unknown source using existing logic
            authority_score, bias_score, source_type = self._assess_unknown_source(
                source_domain, source_platform, metadata
            )
            reliability_score = authority_score  # Default reliability to authority score
            expertise_level = "general"
            specializations = []
            from_authority_database = False

        # Apply adjustments based on metadata quality
        authority_score = self._adjust_for_metadata_quality(authority_score, metadata)

        return {
            "source_identifier": source_domain or f"document_{doc_index}",
            "authority_score": authority_score,
            "bias_score": bias_score,
            "reliability_score": reliability_score,
            "source_type": source_type,
            "expertise_level": expertise_level,
            "specializations": specializations,
            "source_url": source_url,
            "metadata_quality": self._assess_metadata_quality(metadata),
            "from_authority_database": from_authority_database
        }

    def _analyze_specialization_coverage(self, source_details: Dict[str, Any]) -> Dict[str, int]:
        """Analyze what specializations are covered by the sources"""

        specialization_coverage = {}

        for source_info in source_details.values():
            specializations = source_info.get("specializations", [])
            for spec in specializations:
                specialization_coverage[spec] = specialization_coverage.get(spec, 0) + 1

        return specialization_coverage

    def _get_source_type_distribution(self, source_details: Dict[str, Any]) -> Dict[str, int]:
        """Get distribution of source types"""

        type_distribution = {}

        for source_info in source_details.values():
            source_type = source_info.get("source_type", SourceType.USER_GENERATED)
            type_name = source_type.value if hasattr(source_type, 'value') else str(source_type)
            type_distribution[type_name] = type_distribution.get(type_name, 0) + 1

        return type_distribution

    # NOTE: The following methods remain UNCHANGED from the original implementation:
    # - _extract_domain()
    # - _assess_unknown_source()
    # - _adjust_for_metadata_quality()
    # - _assess_metadata_quality()
    # - _create_unverifiable_result()
    # - _create_error_result()

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        if not url:
            return ""

        import re
        # Simple domain extraction
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        return match.group(1).lower() if match else ""

    def _assess_unknown_source(self, domain: str, platform: str, metadata: Dict) -> tuple:
        """Assess authority for unknown sources based on patterns"""

        # Default scores for unknown sources
        authority_score = 0.5
        bias_score = 0.3
        source_type = SourceType.USER_GENERATED

        # Check for government domains
        if domain.endswith('.gov'):
            authority_score = 0.95
            bias_score = 0.05
            source_type = SourceType.REGULATORY

        # Check for educational domains
        elif domain.endswith('.edu'):
            authority_score = 0.85
            bias_score = 0.1
            source_type = SourceType.ACADEMIC

        # Check for manufacturer patterns
        elif any(brand in domain for brand in ['toyota', 'honda', 'ford', 'bmw', 'mercedes']):
            authority_score = 0.90
            bias_score = 0.15
            source_type = SourceType.OFFICIAL

        # Check for known automotive media patterns
        elif any(term in domain for term in ['auto', 'car', 'motor', 'drive']):
            authority_score = 0.65
            bias_score = 0.25
            source_type = SourceType.PROFESSIONAL

        # Check for forum/social media patterns
        elif any(term in domain for term in ['forum', 'reddit', 'facebook', 'twitter']):
            authority_score = 0.35
            bias_score = 0.5
            source_type = SourceType.USER_GENERATED

        return authority_score, bias_score, source_type

    def _adjust_for_metadata_quality(self, base_score: float, metadata: Dict) -> float:
        """Adjust authority score based on metadata quality"""

        # Check for rich metadata (indicates professional source)
        quality_indicators = [
            'author', 'publishedDate', 'title', 'excerpt',
            'vehicleModel', 'modelYear', 'manufacturer'
        ]

        metadata_richness = sum(1 for indicator in quality_indicators if metadata.get(indicator))
        quality_ratio = metadata_richness / len(quality_indicators)

        # Small adjustment based on metadata quality
        adjustment = (quality_ratio - 0.5) * 0.1  # ±5% adjustment

        return max(0.0, min(1.0, base_score + adjustment))

    def _assess_metadata_quality(self, metadata: Dict) -> str:
        """Assess the quality of document metadata"""

        essential_fields = ['url', 'title']
        quality_fields = ['author', 'publishedDate', 'excerpt']
        automotive_fields = ['vehicleModel', 'manufacturer', 'modelYear']

        essential_score = sum(1 for field in essential_fields if metadata.get(field))
        quality_score = sum(1 for field in quality_fields if metadata.get(field))
        automotive_score = sum(1 for field in automotive_fields if metadata.get(field))

        total_score = essential_score + quality_score + automotive_score
        max_score = len(essential_fields) + len(quality_fields) + len(automotive_fields)

        quality_ratio = total_score / max_score

        if quality_ratio >= 0.7:
            return "high"
        elif quality_ratio >= 0.4:
            return "medium"
        else:
            return "low"

    def _create_unverifiable_result(self, start_time: datetime,
                                    precondition_result: PreconditionResult) -> ValidationStepResult:
        """Create result for unverifiable validation"""

        return ValidationStepResult(
            step_id=f"source_credibility_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Source Credibility Analysis",
            status=ValidationStatus.UNVERIFIABLE,
            confidence_impact=0.0,
            summary=precondition_result.failure_reason,
            details={"precondition_failures": [failure.__dict__ for failure in precondition_result.missing_resources]},
            started_at=start_time,
            completed_at=datetime.now(),
            warnings=[
                ValidationWarning(
                    category="precondition_failure",
                    severity="critical",
                    message="Cannot validate source credibility",
                    explanation=precondition_result.failure_reason,
                    suggestion="Check system requirements and data availability"
                )
            ]
        )

    def _create_error_result(self, start_time: datetime, error_message: str) -> ValidationStepResult:
        """Create result for validation errors"""

        return ValidationStepResult(
            step_id=f"source_credibility_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Source Credibility Analysis",
            status=ValidationStatus.FAILED,
            confidence_impact=-10.0,
            summary=f"Validation failed: {error_message}",
            details={"error": error_message},
            started_at=start_time,
            completed_at=datetime.now(),
            warnings=[
                ValidationWarning(
                    category="validation_error",
                    severity="critical",
                    message="Source credibility validation encountered an error",
                    explanation=error_message,
                    suggestion="Check logs and retry validation"
                )
            ]
        )