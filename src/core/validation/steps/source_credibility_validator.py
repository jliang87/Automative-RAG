"""
Source Credibility Validation Step
Evaluates the authority and reliability of information sources
"""

import logging
from typing import Dict, List, Any
import re

from ..models.validation_models import (
    ValidationStepResult, ValidationStatus, ValidationContext,
    create_validation_step_result
)
from ..meta_validator import PreconditionResult
from .base_validation_step import BaseValidationStep

logger = logging.getLogger(__name__)


class SourceCredibilityValidator(BaseValidationStep):
    """
    Validates source credibility and authority
    """

    def __init__(self, config, meta_validator):
        super().__init__(config, meta_validator)
        self.authority_database = self._initialize_authority_database()

    def _initialize_authority_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize source authority database"""

        return {
            # Official automotive sources
            "epa.gov": {
                "authority_score": 1.0,
                "source_type": "official",
                "expertise": "fuel_economy,emissions,testing",
                "bias_score": 0.0,
                "description": "EPA - Official US Environmental Protection Agency"
            },
            "nhtsa.gov": {
                "authority_score": 1.0,
                "source_type": "official",
                "expertise": "safety,recalls,standards",
                "bias_score": 0.0,
                "description": "NHTSA - National Highway Traffic Safety Administration"
            },
            "iihs.org": {
                "authority_score": 0.95,
                "source_type": "official",
                "expertise": "safety,crash_testing",
                "bias_score": 0.0,
                "description": "IIHS - Insurance Institute for Highway Safety"
            },

            # Manufacturer sources
            "toyota.com": {
                "authority_score": 0.9,
                "source_type": "manufacturer",
                "expertise": "specifications,features,pricing",
                "bias_score": 0.3,
                "description": "Toyota Official Website"
            },
            "honda.com": {
                "authority_score": 0.9,
                "source_type": "manufacturer",
                "expertise": "specifications,features,pricing",
                "bias_score": 0.3,
                "description": "Honda Official Website"
            },
            "ford.com": {
                "authority_score": 0.9,
                "source_type": "manufacturer",
                "expertise": "specifications,features,pricing",
                "bias_score": 0.3,
                "description": "Ford Official Website"
            },
            "bmw.com": {
                "authority_score": 0.9,
                "source_type": "manufacturer",
                "expertise": "specifications,features,pricing",
                "bias_score": 0.3,
                "description": "BMW Official Website"
            },
            "mercedes-benz.com": {
                "authority_score": 0.9,
                "source_type": "manufacturer",
                "expertise": "specifications,features,pricing",
                "bias_score": 0.3,
                "description": "Mercedes-Benz Official Website"
            },

            # Professional automotive sources
            "motortrend.com": {
                "authority_score": 0.85,
                "source_type": "professional",
                "expertise": "reviews,testing,industry_news",
                "bias_score": 0.1,
                "description": "Motor Trend - Automotive publication"
            },
            "caranddriver.com": {
                "authority_score": 0.85,
                "source_type": "professional",
                "expertise": "reviews,testing,buying_guides",
                "bias_score": 0.1,
                "description": "Car and Driver Magazine"
            },
            "edmunds.com": {
                "authority_score": 0.8,
                "source_type": "professional",
                "expertise": "reviews,pricing,buying_guides",
                "bias_score": 0.15,
                "description": "Edmunds - Automotive information"
            },
            "autoblog.com": {
                "authority_score": 0.75,
                "source_type": "professional",
                "expertise": "news,reviews,industry_analysis",
                "bias_score": 0.2,
                "description": "Autoblog - Automotive news and reviews"
            },
            "roadandtrack.com": {
                "authority_score": 0.8,
                "source_type": "professional",
                "expertise": "performance,testing,enthusiast",
                "bias_score": 0.15,
                "description": "Road & Track Magazine"
            },

            # Chinese automotive sources
            "autohome.com.cn": {
                "authority_score": 0.7,
                "source_type": "professional",
                "expertise": "chinese_market,reviews,specifications",
                "bias_score": 0.2,
                "description": "Autohome - Chinese automotive platform"
            },
            "pcauto.com.cn": {
                "authority_score": 0.65,
                "source_type": "professional",
                "expertise": "chinese_market,news,reviews",
                "bias_score": 0.25,
                "description": "PCauto - Chinese automotive website"
            },

            # Video platforms
            "youtube.com": {
                "authority_score": 0.4,
                "source_type": "user_generated",
                "expertise": "varies",
                "bias_score": 0.4,
                "description": "YouTube - User-generated video content"
            },
            "bilibili.com": {
                "authority_score": 0.35,
                "source_type": "user_generated",
                "expertise": "chinese_content,varies",
                "bias_score": 0.4,
                "description": "Bilibili - Chinese video platform"
            },

            # Forums and community
            "reddit.com": {
                "authority_score": 0.3,
                "source_type": "user_generated",
                "expertise": "community_opinions,experiences",
                "bias_score": 0.5,
                "description": "Reddit - Community discussions"
            },
            "forums.": {  # Generic forum pattern
                "authority_score": 0.25,
                "source_type": "user_generated",
                "expertise": "community_experiences",
                "bias_score": 0.6,
                "description": "Automotive forums"
            }
        }

    async def _execute_validation(
            self,
            context: ValidationContext,
            precondition_result: PreconditionResult
    ) -> ValidationStepResult:
        """Execute source credibility validation"""

        documents = context.documents

        if not documents:
            return create_validation_step_result(
                step_type=self.step_type,
                status=ValidationStatus.FAILED,
                summary="No documents available for credibility assessment",
                confidence_impact=0.0
            )

        # Analyze source credibility
        credibility_analysis = self._analyze_source_credibility(documents)

        # Determine validation status and confidence impact
        status, confidence_impact = self._determine_credibility_status(credibility_analysis)

        # Create result
        result = create_validation_step_result(
            step_type=self.step_type,
            status=status,
            summary=self._create_credibility_summary(credibility_analysis),
            confidence_impact=confidence_impact
        )

        # Add detailed analysis
        result.details = credibility_analysis
        result.sources_used = [doc.get('metadata', {}).get('url', 'unknown') for doc in documents]

        # Add warnings if needed
        self._add_credibility_warnings(result, credibility_analysis)

        return result

    def _analyze_source_credibility(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze credibility of all sources"""

        source_analyses = []
        authority_scores = []
        bias_scores = []
        source_types = set()

        for doc in documents:
            analysis = self._analyze_single_source(doc)
            source_analyses.append(analysis)

            if analysis['authority_score'] is not None:
                authority_scores.append(analysis['authority_score'])
            if analysis['bias_score'] is not None:
                bias_scores.append(analysis['bias_score'])
            if analysis['source_type']:
                source_types.add(analysis['source_type'])

        # Calculate aggregate metrics
        avg_authority = sum(authority_scores) / len(authority_scores) if authority_scores else 0.0
        avg_bias = sum(bias_scores) / len(bias_scores) if bias_scores else 0.5

        # Count source types
        official_count = sum(1 for s in source_analyses if s['source_type'] == 'official')
        professional_count = sum(1 for s in source_analyses if s['source_type'] == 'professional')
        user_generated_count = sum(1 for s in source_analyses if s['source_type'] == 'user_generated')

        # Assess source diversity
        diversity_score = self._calculate_diversity_score(source_types)

        return {
            "total_sources": len(documents),
            "source_analyses": source_analyses,
            "avg_authority_score": avg_authority,
            "avg_bias_score": avg_bias,
            "source_type_counts": {
                "official": official_count,
                "professional": professional_count,
                "user_generated": user_generated_count
            },
            "source_type_diversity": len(source_types),
            "diversity_score": diversity_score,
            "highest_authority_source": max(source_analyses,
                                            key=lambda x: x['authority_score'] or 0) if source_analyses else None,
            "bias_detected": avg_bias > 0.4,
            "authority_threshold_met": avg_authority >= 0.6
        }

    def _analyze_single_source(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze credibility of a single source"""

        metadata = document.get('metadata', {})

        # Extract URL or source identifier
        url = metadata.get('url', metadata.get('source', ''))
        domain = self._extract_domain(url)

        # Get source information from database
        source_info = self._get_source_info(domain, url)

        # Extract additional metadata
        source_type = metadata.get('source_type',
                                   metadata.get('sourcePlatform', source_info.get('source_type', 'unknown')))
        author = metadata.get('author', metadata.get('uploader', metadata.get('authorName', '')))

        # Calculate freshness score
        freshness_score = self._calculate_freshness_score(metadata)

        analysis = {
            "url": url,
            "domain": domain,
            "source_type": source_type,
            "authority_score": source_info.get('authority_score'),
            "bias_score": source_info.get('bias_score'),
            "author": author,
            "freshness_score": freshness_score,
            "expertise_match": self._assess_expertise_match(source_info, metadata),
            "database_match": domain in self.authority_database,
            "source_description": source_info.get('description', ''),
            "credibility_issues": []
        }

        # Identify credibility issues
        self._identify_credibility_issues(analysis)

        return analysis

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""

        if not url:
            return ""

        # Simple domain extraction
        if "://" in url:
            url = url.split("://")[1]

        domain = url.split("/")[0].lower()

        # Remove www prefix
        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    def _get_source_info(self, domain: str, url: str) -> Dict[str, Any]:
        """Get source information from authority database"""

        # Direct domain match
        if domain in self.authority_database:
            return self.authority_database[domain]

        # Pattern matching for generic patterns
        for pattern, info in self.authority_database.items():
            if pattern.endswith(".") and domain.startswith(pattern[:-1]):
                return info

        # Default for unknown sources
        return {
            "authority_score": 0.2,
            "source_type": "unknown",
            "bias_score": 0.5,
            "description": f"Unknown source: {domain}"
        }

    def _calculate_freshness_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate freshness score based on publication date"""

        from datetime import datetime

        # Try different date fields
        date_fields = ['publishedDate', 'upload_date', 'created_at', 'date']

        for field in date_fields:
            date_value = metadata.get(field)
            if date_value:
                try:
                    if isinstance(date_value, str):
                        # Handle different date formats
                        if len(date_value) >= 4 and date_value[:4].isdigit():
                            year = int(date_value[:4])
                            current_year = datetime.now().year
                            age = current_year - year

                            # Freshness decays 10% per year
                            freshness = max(0.0, 1.0 - (age * 0.1))
                            return freshness

                except (ValueError, TypeError):
                    continue

        # Default freshness for unknown dates
        return 0.5

    def _assess_expertise_match(self, source_info: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        """Assess how well source expertise matches query domain"""

        expertise = source_info.get('expertise', '')
        if not expertise:
            return 0.5

        # Check if source has automotive expertise
        automotive_keywords = ['automotive', 'cars', 'vehicles', 'specifications', 'reviews']
        expertise_areas = expertise.split(',')

        match_score = 0.0
        for area in expertise_areas:
            area = area.strip().lower()
            if any(keyword in area for keyword in automotive_keywords):
                match_score = 1.0
                break
            elif area in ['fuel_economy', 'safety', 'emissions', 'testing']:
                match_score = 1.0
                break
            elif area in ['news', 'industry_analysis']:
                match_score = 0.8
            elif area == 'varies':
                match_score = 0.3

        return match_score

    def _identify_credibility_issues(self, analysis: Dict[str, Any]):
        """Identify potential credibility issues"""

        issues = []

        # Low authority score
        if analysis['authority_score'] and analysis['authority_score'] < 0.3:
            issues.append("Low source authority score")

        # High bias score
        if analysis['bias_score'] and analysis['bias_score'] > 0.6:
            issues.append("High potential bias detected")

        # Old content
        if analysis['freshness_score'] < 0.3:
            issues.append("Content may be outdated")

        # No expertise match
        if analysis['expertise_match'] < 0.3:
            issues.append("Source expertise may not match automotive domain")

        # Unknown source
        if not analysis['database_match']:
            issues.append("Source not in authority database")

        analysis['credibility_issues'] = issues

    def _calculate_diversity_score(self, source_types: set) -> float:
        """Calculate source diversity score"""

        type_count = len(source_types)

        if type_count >= 3:
            return 1.0  # Excellent diversity
        elif type_count == 2:
            return 0.7  # Good diversity
        elif type_count == 1:
            return 0.3  # Limited diversity
        else:
            return 0.0  # No diversity

    def _determine_credibility_status(self, analysis: Dict[str, Any]) -> tuple[ValidationStatus, float]:
        """Determine validation status and confidence impact"""

        avg_authority = analysis['avg_authority_score']
        diversity_score = analysis['diversity_score']
        official_count = analysis['source_type_counts']['official']
        bias_detected = analysis['bias_detected']

        # Calculate base confidence impact
        confidence_impact = 0.0

        # Authority contribution
        if avg_authority >= 0.8:
            confidence_impact += 15.0
            status = ValidationStatus.PASSED
        elif avg_authority >= 0.6:
            confidence_impact += 10.0
            status = ValidationStatus.PASSED
        elif avg_authority >= 0.4:
            confidence_impact += 5.0
            status = ValidationStatus.WARNING
        else:
            confidence_impact -= 5.0
            status = ValidationStatus.WARNING

        # Diversity bonus
        confidence_impact += diversity_score * 8.0

        # Official source bonus
        confidence_impact += min(10.0, official_count * 5.0)

        # Bias penalty
        if bias_detected:
            confidence_impact -= 8.0
            if status == ValidationStatus.PASSED:
                status = ValidationStatus.WARNING

        # Final status adjustment
        if confidence_impact < -5.0:
            status = ValidationStatus.FAILED

        return status, confidence_impact

    def _create_credibility_summary(self, analysis: Dict[str, Any]) -> str:
        """Create human-readable summary of credibility analysis"""

        total_sources = analysis['total_sources']
        avg_authority = analysis['avg_authority_score']
        official_count = analysis['source_type_counts']['official']
        professional_count = analysis['source_type_counts']['professional']
        diversity = analysis['source_type_diversity']

        if avg_authority >= 0.8:
            authority_desc = "high authority"
        elif avg_authority >= 0.6:
            authority_desc = "good authority"
        elif avg_authority >= 0.4:
            authority_desc = "moderate authority"
        else:
            authority_desc = "limited authority"

        summary_parts = [
            f"Assessed {total_sources} sources with {authority_desc}"
        ]

        if official_count > 0:
            summary_parts.append(f"{official_count} official source(s)")

        if professional_count > 0:
            summary_parts.append(f"{professional_count} professional source(s)")

        summary_parts.append(f"{diversity} source type(s)")

        return ", ".join(summary_parts)

    def _add_credibility_warnings(self, result: ValidationStepResult, analysis: Dict[str, Any]):
        """Add credibility-related warnings"""

        # Low authority warning
        if analysis['avg_authority_score'] < 0.5:
            self._add_warning(
                result,
                category="low_authority",
                severity="caution",
                message="Sources have limited authority",
                explanation=f"Average authority score: {analysis['avg_authority_score']:.2f}",
                suggestion="Seek additional authoritative sources"
            )

        # Bias warning
        if analysis['bias_detected']:
            self._add_warning(
                result,
                category="bias_detected",
                severity="caution",
                message="Potential bias detected in sources",
                explanation=f"Average bias score: {analysis['avg_bias_score']:.2f}",
                suggestion="Consider sources with different perspectives"
            )

        # Low diversity warning
        if analysis['diversity_score'] < 0.5:
            self._add_warning(
                result,
                category="low_diversity",
                severity="info",
                message="Limited source diversity",
                explanation=f"Only {analysis['source_type_diversity']} source type(s)",
                suggestion="Include sources from different types (official, professional, community)"
            )