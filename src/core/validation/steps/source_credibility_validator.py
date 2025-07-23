"""
Source Credibility Validator
Validates source authority and credibility for automotive information
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.models.schema import (
    ValidationStepResult, ValidationStatus, ValidationStepType,
    ValidationContext, ValidationWarning, SourceType
)
from .steps_readiness_checker import MetaValidator, PreconditionResult

logger = logging.getLogger(__name__)


class SourceCredibilityValidator:
    """
    Validates the credibility and authority of information sources
    """

    def __init__(self, step_config: Dict[str, Any], meta_validator: MetaValidator):
        self.step_config = step_config
        self.meta_validator = meta_validator
        self.step_type = ValidationStepType.SOURCE_CREDIBILITY

        # Authority scoring database (in production, this would be a real database)
        self.authority_scores = self._initialize_authority_database()

    def _initialize_authority_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize source authority scoring database"""
        return {
            # Official Sources (High Authority)
            "epa.gov": {"authority": 1.0, "bias": 0.0, "type": SourceType.REGULATORY},
            "nhtsa.gov": {"authority": 1.0, "bias": 0.0, "type": SourceType.REGULATORY},
            "toyota.com": {"authority": 0.95, "bias": 0.1, "type": SourceType.OFFICIAL},
            "honda.com": {"authority": 0.95, "bias": 0.1, "type": SourceType.OFFICIAL},
            "ford.com": {"authority": 0.95, "bias": 0.1, "type": SourceType.OFFICIAL},
            "bmw.com": {"authority": 0.95, "bias": 0.1, "type": SourceType.OFFICIAL},
            "mercedes-benz.com": {"authority": 0.95, "bias": 0.1, "type": SourceType.OFFICIAL},

            # Professional Automotive Media (Medium-High Authority)
            "caranddriver.com": {"authority": 0.85, "bias": 0.2, "type": SourceType.PROFESSIONAL},
            "motortrend.com": {"authority": 0.85, "bias": 0.2, "type": SourceType.PROFESSIONAL},
            "roadandtrack.com": {"authority": 0.80, "bias": 0.2, "type": SourceType.PROFESSIONAL},
            "automobilemag.com": {"authority": 0.80, "bias": 0.2, "type": SourceType.PROFESSIONAL},
            "autoweek.com": {"authority": 0.75, "bias": 0.2, "type": SourceType.PROFESSIONAL},
            "kbb.com": {"authority": 0.85, "bias": 0.15, "type": SourceType.PROFESSIONAL},
            "edmunds.com": {"authority": 0.85, "bias": 0.15, "type": SourceType.PROFESSIONAL},

            # Consumer/Review Sites (Medium Authority)
            "consumerreports.org": {"authority": 0.90, "bias": 0.1, "type": SourceType.PROFESSIONAL},
            "carsguide.com.au": {"authority": 0.70, "bias": 0.3, "type": SourceType.PROFESSIONAL},
            "cars.com": {"authority": 0.70, "bias": 0.25, "type": SourceType.PROFESSIONAL},

            # User-Generated Content (Lower Authority)
            "reddit.com": {"authority": 0.40, "bias": 0.4, "type": SourceType.USER_GENERATED},
            "forums.edmunds.com": {"authority": 0.45, "bias": 0.4, "type": SourceType.USER_GENERATED},
            "carsguru.net": {"authority": 0.35, "bias": 0.5, "type": SourceType.USER_GENERATED},

            # Academic/Research (High Authority)
            "sae.org": {"authority": 0.95, "bias": 0.05, "type": SourceType.ACADEMIC},
            "ieee.org": {"authority": 0.95, "bias": 0.05, "type": SourceType.ACADEMIC},
        }

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
        """Perform the actual source credibility validation"""

        documents = context.documents
        warnings = []
        sources_analyzed = []

        # Analyze each document's source credibility
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

        # Calculate overall source credibility metrics
        avg_authority = sum(source_scores) / len(source_scores) if source_scores else 0.0
        source_diversity = len(set(source_details[src]["source_type"] for src in source_details))
        official_source_count = sum(1 for src in source_details.values()
                                    if src["source_type"] == SourceType.OFFICIAL)

        # Determine validation status
        if avg_authority >= 0.8 and source_diversity >= 2:
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

        # Build summary
        summary = (f"Analyzed {len(documents)} sources. "
                   f"Average authority: {avg_authority:.2f}/1.0, "
                   f"Source diversity: {source_diversity} types, "
                   f"Official sources: {official_source_count}")

        # Build detailed results
        details = {
            "sources_analyzed": len(documents),
            "avg_authority_score": avg_authority,
            "source_type_diversity": source_diversity,
            "official_sources_used": official_source_count,
            "source_breakdown": source_details,
            "authority_distribution": {
                "high": sum(1 for s in source_scores if s >= 0.8),
                "medium": sum(1 for s in source_scores if 0.5 <= s < 0.8),
                "low": sum(1 for s in source_scores if s < 0.5)
            },
            "bias_detected": any(src["bias_score"] > 0.4 for src in source_details.values())
        }

        return ValidationStepResult(
            step_id=f"source_credibility_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Source Credibility Analysis",
            status=status,
            confidence_impact=confidence_impact,
            summary=summary,
            details=details,
            started_at=datetime.now(),
            warnings=warnings,
            sources_used=sources_analyzed
        )

    def _analyze_document_source(self, doc: Dict[str, Any], doc_index: int) -> Dict[str, Any]:
        """Analyze the credibility of a single document's source"""

        metadata = doc.get("metadata", {})

        # Extract source information
        source_url = metadata.get("url", "")
        source_domain = self._extract_domain(source_url)
        source_platform = metadata.get("sourcePlatform", metadata.get("source", "unknown"))

        # Look up in authority database
        authority_data = self.authority_scores.get(source_domain, {})

        if authority_data:
            # Known source
            authority_score = authority_data["authority"]
            bias_score = authority_data["bias"]
            source_type = authority_data["type"]
        else:
            # Unknown source - assess based on patterns
            authority_score, bias_score, source_type = self._assess_unknown_source(
                source_domain, source_platform, metadata
            )

        # Apply adjustments based on metadata quality
        authority_score = self._adjust_for_metadata_quality(authority_score, metadata)

        return {
            "source_identifier": source_domain or f"document_{doc_index}",
            "authority_score": authority_score,
            "bias_score": bias_score,
            "source_type": source_type,
            "source_url": source_url,
            "metadata_quality": self._assess_metadata_quality(metadata)
        }

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