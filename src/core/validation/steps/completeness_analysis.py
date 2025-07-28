"""
Consensus Analysis Validator - COMPLETE UPDATED VERSION
Analyzes consensus across multiple sources using model-driven claim analysis
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from collections import Counter, defaultdict

from src.models import (
    ValidationStepResult, ValidationStatus, ValidationStepType,
    ValidationContext, ValidationWarning
)
from src.models.knowledge_models import (
    FactualClaim, ConsensusAnalysisResult, SourceAuthority, SourceType
)
from .steps_readiness_checker import MetaValidator, PreconditionResult

logger = logging.getLogger(__name__)


class ConsensusValidator:
    """
    Analyzes consensus across multiple sources using model-driven claim analysis
    """

    def __init__(self,
                 step_config: Dict[str, Any],
                 meta_validator: MetaValidator,
                 source_authorities: Optional[List[SourceAuthority]] = None):
        self.step_config = step_config
        self.meta_validator = meta_validator
        self.step_type = ValidationStepType.CONSENSUS

        # Build authority lookup from model data instead of hardcoded
        self.authority_lookup = {}
        if source_authorities:
            for auth in source_authorities:
                self.authority_lookup[auth.domain] = auth
        else:
            # Load default authorities if none provided
            self._load_default_authorities()

        # Define what constitutes factual claims for automotive content
        self.factual_claim_patterns = self._initialize_claim_patterns()

    def _load_default_authorities(self):
        """Load default source authorities for consensus analysis"""
        default_authorities = [
            SourceAuthority(
                domain="epa.gov",
                source_type=SourceType.REGULATORY,
                authority_score=1.0,
                bias_score=0.0,
                expertise_level="expert",
                specializations=["fuel_economy", "emissions"]
            ),
            SourceAuthority(
                domain="nhtsa.gov",
                source_type=SourceType.REGULATORY,
                authority_score=1.0,
                bias_score=0.0,
                expertise_level="expert",
                specializations=["safety", "crash_testing"]
            ),
            SourceAuthority(
                domain="toyota.com",
                source_type=SourceType.OFFICIAL,
                authority_score=0.95,
                bias_score=0.1,
                expertise_level="expert",
                specializations=["toyota"]
            ),
            SourceAuthority(
                domain="honda.com",
                source_type=SourceType.OFFICIAL,
                authority_score=0.95,
                bias_score=0.1,
                expertise_level="expert",
                specializations=["honda"]
            ),
            SourceAuthority(
                domain="caranddriver.com",
                source_type=SourceType.PROFESSIONAL,
                authority_score=0.85,
                bias_score=0.2,
                expertise_level="professional",
                specializations=["reviews", "testing"]
            ),
            SourceAuthority(
                domain="motortrend.com",
                source_type=SourceType.PROFESSIONAL,
                authority_score=0.85,
                bias_score=0.2,
                expertise_level="professional",
                specializations=["reviews", "testing"]
            ),
            SourceAuthority(
                domain="edmunds.com",
                source_type=SourceType.PROFESSIONAL,
                authority_score=0.85,
                bias_score=0.15,
                expertise_level="professional",
                specializations=["pricing", "reviews"]
            ),
            SourceAuthority(
                domain="reddit.com",
                source_type=SourceType.USER_GENERATED,
                authority_score=0.40,
                bias_score=0.4,
                expertise_level="enthusiast",
                specializations=["discussions"]
            )
        ]

        for auth in default_authorities:
            self.authority_lookup[auth.domain] = auth

    def _initialize_claim_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for extracting factual claims"""
        return {
            "fuel_economy": [
                r'(\d+)\s*mpg',
                r'(\d+)\s*miles per gallon',
                r'fuel economy.*?(\d+)',
                r'gas mileage.*?(\d+)'
            ],
            "performance": [
                r'(\d+)\s*(?:hp|horsepower)',
                r'(\d+)\s*(?:lb-ft|lbft|torque)',
                r'0-60.*?(\d+\.?\d*)\s*sec',
                r'quarter mile.*?(\d+\.?\d*)\s*sec'
            ],
            "pricing": [
                r'\$(\d+,?\d+)',
                r'price.*?\$(\d+,?\d+)',
                r'cost.*?\$(\d+,?\d+)',
                r'msrp.*?\$(\d+,?\d+)'
            ],
            "safety": [
                r'(\d+)\s*star',
                r'nhtsa.*?(\d+)',
                r'iihs.*?(top safety pick)',
                r'safety rating.*?(\d+)'
            ],
            "dimensions": [
                r'(\d+)\s*inches?\s*long',
                r'(\d+)\s*inches?\s*wide',
                r'(\d+,?\d+)\s*(?:lbs?|pounds)',
                r'cargo.*?(\d+\.?\d*)\s*cubic feet'
            ],
            "reliability": [
                r'(\d+)\s*(?:years?|yr)\s*warranty',
                r'reliability.*?(\d+)/10',
                r'jd power.*?(\d+)',
                r'problems per 100.*?(\d+)'
            ]
        }

    async def execute(self, context: ValidationContext) -> ValidationStepResult:
        """Execute consensus analysis"""

        start_time = datetime.now()

        # Check preconditions
        precondition_result = await self.meta_validator.check_preconditions(
            self.step_type, context, self.step_config
        )

        if precondition_result.status != "READY":
            return self._create_unverifiable_result(start_time, precondition_result)

        try:
            # Perform consensus analysis
            result = await self._perform_validation(context)
            result.completed_at = datetime.now()
            result.duration_ms = int((result.completed_at - start_time).total_seconds() * 1000)

            return result

        except Exception as e:
            logger.error(f"Consensus analysis failed: {str(e)}")
            return self._create_error_result(start_time, str(e))

    async def _perform_validation(self, context: ValidationContext) -> ValidationStepResult:
        """Perform the actual consensus analysis using model-driven approach"""

        documents = context.documents
        warnings = []
        sources_analyzed = []

        # Extract claims from all documents using model structures
        all_claims = []
        for i, doc in enumerate(documents):
            doc_claims = self._extract_claims_from_document(doc, i)
            all_claims.extend(doc_claims)
            sources_analyzed.append(f"document_{i}")

        # Group claims by category and analyze consensus using models
        consensus_results = {}
        for category in self.factual_claim_patterns.keys():
            category_claims = [claim for claim in all_claims if claim.claim_category == category]
            if category_claims:
                consensus_results[f"{category}_consensus"] = self._analyze_category_consensus_with_models(
                    category_claims, category
                )

        # Analyze source diversity and reliability using model data
        source_analysis = self._analyze_source_diversity_with_models(documents)

        # Calculate overall consensus metrics using model-based analysis
        consensus_metrics = self._calculate_consensus_metrics_with_models(consensus_results, source_analysis)

        # Generate warnings for significant disagreements
        for category, analysis in consensus_results.items():
            if analysis.consensus_strength < 0.7 and len(analysis.claims) >= 2:
                disagreements = analysis.disagreements
                if disagreements:
                    severity = "critical" if analysis.consensus_strength < 0.5 else "caution"
                    warnings.append(ValidationWarning(
                        category=f"consensus_{category}",
                        severity=severity,
                        message=f"Sources disagree on {category.replace('_', ' ')}",
                        explanation=f"Consensus strength: {analysis.consensus_strength:.1%}. Disagreements: {len(disagreements)}",
                        suggestion="Cross-reference with authoritative sources"
                    ))

        # Check for source diversity issues
        if source_analysis["diversity_score"] < 0.6:
            warnings.append(ValidationWarning(
                category="source_diversity",
                severity="caution",
                message="Limited source diversity may affect consensus reliability",
                explanation=f"Source diversity score: {source_analysis['diversity_score']:.1%}",
                suggestion="Include sources from different types (official, professional, user-generated)"
            ))

        # Determine validation status and confidence impact
        overall_consensus = consensus_metrics["overall_consensus_strength"]
        source_diversity = source_analysis["diversity_score"]

        # Weight consensus and diversity
        weighted_score = (overall_consensus * 0.7) + (source_diversity * 0.3)

        if weighted_score >= 0.8:
            status = ValidationStatus.PASSED
            confidence_impact = 12.0 + (weighted_score - 0.8) * 15  # 12-15 point boost
        elif weighted_score >= 0.6:
            status = ValidationStatus.WARNING
            confidence_impact = 5.0 + (weighted_score - 0.6) * 35  # 5-12 point boost
        else:
            status = ValidationStatus.WARNING
            confidence_impact = max(-3.0, (weighted_score - 0.3) * 10)  # Up to -3 point penalty

            warnings.append(ValidationWarning(
                category="overall_consensus",
                severity="critical",
                message="Low consensus across sources",
                explanation=f"Overall consensus strength: {overall_consensus:.1%}",
                suggestion="Seek additional authoritative sources to resolve disagreements"
            ))

        # Build summary
        total_claims = len(all_claims)
        total_agreements = sum(
            len([claim for claim in analysis.claims if claim.confidence > 0.7])
            for analysis in consensus_results.values()
        )

        summary = (f"Analyzed {total_claims} claims across {len(documents)} sources. "
                   f"Overall consensus: {overall_consensus:.1%}, "
                   f"Source diversity: {source_diversity:.1%}, "
                   f"High-confidence claims: {total_agreements}/{total_claims}")

        # Build detailed results
        details = {
            "sources_analyzed": len(documents),
            "total_claims_found": total_claims,
            "total_agreements": total_agreements,
            "overall_consensus_strength": overall_consensus,
            "source_diversity_score": source_diversity,
            "consensus_by_category": {
                category: {
                    "consensus_strength": analysis.consensus_strength,
                    "total_claims": analysis.total_claims,
                    "agreements": len([claim for claim in analysis.claims if claim.confidence > 0.7]),
                    "disagreements": len(analysis.disagreements),
                    "consensus_value": analysis.consensus_value
                }
                for category, analysis in consensus_results.items()
            },
            "source_analysis": source_analysis,
            "consensus_metrics": consensus_metrics,
            "claim_distribution": self._get_claim_distribution_with_models(all_claims),
            "disagreement_summary": self._summarize_disagreements_with_models(consensus_results),
            "authority_scores_used": {
                domain: auth.authority_score
                for domain, auth in self.authority_lookup.items()
            }
        }

        return ValidationStepResult(
            step_id=f"consensus_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Consensus Analysis",
            status=status,
            confidence_impact=confidence_impact,
            summary=summary,
            details=details,
            started_at=start_time,
            warnings=warnings,
            sources_used=sources_analyzed
        )

    def _extract_claims_from_document(self, doc: Dict[str, Any], doc_index: int) -> List[FactualClaim]:
        """Extract factual claims from a document using FactualClaim models"""

        content = doc.get("content", "")
        metadata = doc.get("metadata", {})

        claims = []
        source_id = metadata.get("url", f"document_{doc_index}")
        source_domain = self._extract_domain(source_id)

        # Get authority information from models
        source_authority_obj = self.authority_lookup.get(source_domain)
        if source_authority_obj:
            source_authority = source_authority_obj.authority_score
            source_type = source_authority_obj.source_type
        else:
            source_authority = 0.5  # Default for unknown sources
            source_type = SourceType.USER_GENERATED

        # Extract claims for each category using existing patterns but create model instances
        for category, patterns in self.factual_claim_patterns.items():
            category_claims = self._extract_category_claims(content, category, patterns)
            for claim_data in category_claims:
                claim = FactualClaim(
                    claim_id=f"{source_domain}_{category}_{len(claims)}",
                    claim_type=claim_data["type"],
                    claim_category=category,
                    claim_text=claim_data["raw_text"],
                    extracted_value=claim_data["value"],
                    context=content[max(0, claim_data.get("start_pos", 0) - 50):
                            claim_data.get("end_pos", 0) + 50] if claim_data.get("start_pos") else claim_data["raw_text"],
                    confidence=1.0,  # Initial confidence, will be adjusted based on consensus
                    source_id=source_id,
                    source_type=source_type,
                    source_authority=source_authority,
                    document_index=doc_index
                )
                claims.append(claim)

        return claims

    def _extract_category_claims(self, content: str, category: str, patterns: List[str]) -> List[Dict[str, Any]]:
        """Extract claims for a specific category"""

        claims = []
        content_lower = content.lower()

        for pattern in patterns:
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                # Extract the numerical value or text
                if match.groups():
                    value = match.group(1)
                    # Try to convert to number if possible
                    try:
                        if ',' in value:
                            value = value.replace(',', '')
                        numeric_value = float(value)
                        claim_value = numeric_value
                    except ValueError:
                        claim_value = value.strip()
                else:
                    claim_value = match.group(0)

                # Determine claim type based on pattern
                claim_type = self._determine_claim_type(pattern, category)

                claims.append({
                    "type": claim_type,
                    "value": claim_value,
                    "raw_text": match.group(0),
                    "pattern_used": pattern,
                    "start_pos": match.start(),
                    "end_pos": match.end()
                })

        return claims

    def _analyze_category_consensus_with_models(self, claims: List[FactualClaim], category: str) -> ConsensusAnalysisResult:
        """Analyze consensus for a specific category using model structures"""

        if len(claims) < 2:
            return ConsensusAnalysisResult(
                category=category,
                total_claims=len(claims),
                consensus_strength=1.0,  # No disagreement if only one claim
                agreement_count=len(claims),
                disagreement_count=0,
                claims=claims,
                consensus_value=claims[0].extracted_value if claims else None,
                disagreements=[],
                source_diversity=1.0,
                authority_weighted_consensus=1.0
            )

        # Group claims by type
        claims_by_type = defaultdict(list)
        for claim in claims:
            claims_by_type[claim.claim_type].append(claim)

        agreements = 0
        total_comparisons = 0
        disagreements = []
        consensus_values = {}

        # Analyze each claim type
        for claim_type, type_claims in claims_by_type.items():
            if len(type_claims) < 2:
                continue

            type_analysis = self._analyze_claim_type_consensus_with_models(type_claims, claim_type)

            agreements += type_analysis["agreements"]
            total_comparisons += type_analysis["total_comparisons"]
            disagreements.extend(type_analysis["disagreements"])

            if type_analysis["consensus_value"] is not None:
                consensus_values[claim_type] = type_analysis["consensus_value"]

        # Calculate consensus strength
        if total_comparisons > 0:
            consensus_strength = agreements / total_comparisons
        else:
            consensus_strength = 1.0

        # Calculate source diversity
        source_types = set(claim.source_type for claim in claims)
        source_diversity = len(source_types) / len(SourceType) if len(SourceType) > 0 else 1.0

        # Calculate authority-weighted consensus
        total_authority = sum(claim.source_authority for claim in claims)
        authority_weighted_consensus = (
            sum(claim.source_authority for claim in claims if claim.confidence > 0.7) / total_authority
            if total_authority > 0 else 0.0
        )

        # Determine overall consensus value
        consensus_value = None
        if consensus_values:
            # Use the consensus value from the most common claim type
            most_common_type = max(claims_by_type.keys(), key=lambda k: len(claims_by_type[k]))
            consensus_value = consensus_values.get(most_common_type)

        return ConsensusAnalysisResult(
            category=category,
            total_claims=len(claims),
            consensus_strength=consensus_strength,
            agreement_count=agreements,
            disagreement_count=len(disagreements),
            claims=claims,
            consensus_value=consensus_value,
            disagreements=disagreements,
            source_diversity=source_diversity,
            authority_weighted_consensus=authority_weighted_consensus
        )

    def _analyze_claim_type_consensus_with_models(self, claims: List[FactualClaim], claim_type: str) -> Dict[str, Any]:
        """Analyze consensus for a specific type of claim using models"""

        if len(claims) < 2:
            return {
                "agreements": 0,
                "total_comparisons": 0,
                "disagreements": [],
                "consensus_value": claims[0].extracted_value if claims else None
            }

        # Check for agreements and disagreements
        agreements = 0
        total_comparisons = 0
        disagreements = []

        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                total_comparisons += 1

                claim1, claim2 = claims[i], claims[j]
                val1, val2 = claim1.extracted_value, claim2.extracted_value

                if self._values_agree(val1, val2, claim_type):
                    agreements += 1
                else:
                    disagreements.append({
                        "claim_type": claim_type,
                        "value1": val1,
                        "value2": val2,
                        "source1": claim1.source_id,
                        "source2": claim2.source_id,
                        "authority1": claim1.source_authority,
                        "authority2": claim2.source_authority,
                        "difference": self._calculate_difference(val1, val2),
                        "claim1_id": claim1.claim_id,
                        "claim2_id": claim2.claim_id
                    })

        # Determine consensus value using authority weighting
        consensus_value = self._determine_consensus_value_with_models(claims, claim_type)

        return {
            "agreements": agreements,
            "total_comparisons": total_comparisons,
            "disagreements": disagreements,
            "consensus_value": consensus_value
        }

    def _determine_consensus_value_with_models(self, claims: List[FactualClaim], claim_type: str) -> Any:
        """Determine the consensus value from multiple claims using model authority data"""

        if not claims:
            return None

        # Weight by source authority from models
        weighted_values = [(claim.extracted_value, claim.source_authority) for claim in claims]

        # For numerical values, calculate weighted average
        try:
            numerical_values = [(float(val), weight) for val, weight in weighted_values]
            total_weight = sum(weight for _, weight in numerical_values)

            if total_weight > 0:
                weighted_sum = sum(val * weight for val, weight in numerical_values)
                return weighted_sum / total_weight
            else:
                return numerical_values[0][0] if numerical_values else None

        except (ValueError, TypeError):
            # For non-numerical values, return most authoritative
            max_authority = max(claim.source_authority for claim in claims)
            authoritative_claims = [
                claim for claim in claims
                if claim.source_authority == max_authority
            ]

            # Return most common among most authoritative
            authoritative_values = [claim.extracted_value for claim in authoritative_claims]
            return Counter(authoritative_values).most_common(1)[0][0] if authoritative_values else None

    def _analyze_source_diversity_with_models(self, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze diversity of sources using model authority data"""

        source_types = []
        source_authorities = []
        source_domains = set()

        for doc in documents:
            metadata = doc.get("metadata", {})
            url = metadata.get("url", "")
            domain = self._extract_domain(url)

            if domain in self.authority_lookup:
                auth_obj = self.authority_lookup[domain]
                source_type = auth_obj.source_type
                source_authority = auth_obj.authority_score
            else:
                source_type = SourceType.USER_GENERATED
                source_authority = 0.5

            source_types.append(source_type)
            source_authorities.append(source_authority)
            if domain:
                source_domains.add(domain)

        # Calculate diversity metrics
        unique_types = set(source_types)
        type_diversity = len(unique_types) / max(len(source_types), 1)

        domain_diversity = len(source_domains) / max(len(documents), 1)

        authority_variance = self._calculate_authority_variance(source_authorities)

        # Overall diversity score
        diversity_score = (type_diversity * 0.4) + (domain_diversity * 0.4) + (authority_variance * 0.2)

        return {
            "diversity_score": diversity_score,
            "type_diversity": type_diversity,
            "domain_diversity": domain_diversity,
            "authority_variance": authority_variance,
            "source_types": list(unique_types),
            "source_type_distribution": Counter([st.value for st in source_types]),
            "unique_domains": len(source_domains),
            "authority_range": {
                "min": min(source_authorities) if source_authorities else 0,
                "max": max(source_authorities) if source_authorities else 0,
                "avg": sum(source_authorities) / len(source_authorities) if source_authorities else 0
            }
        }

    def _calculate_consensus_metrics_with_models(self, consensus_results: Dict[str, ConsensusAnalysisResult],
                                               source_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall consensus metrics using model-based analysis"""

        # Calculate weighted consensus across categories
        category_weights = {
            "fuel_economy_consensus": 0.25,
            "performance_consensus": 0.20,
            "pricing_consensus": 0.15,
            "safety_consensus": 0.20,
            "dimensions_consensus": 0.10,
            "reliability_consensus": 0.10
        }

        weighted_consensus = 0.0
        total_weight = 0.0

        for category, analysis in consensus_results.items():
            if analysis.total_claims > 0:  # Only include categories with claims
                weight = category_weights.get(category, 0.1)
                weighted_consensus += analysis.consensus_strength * weight
                total_weight += weight

        overall_consensus = weighted_consensus / total_weight if total_weight > 0 else 1.0

        # Count total disagreements
        total_disagreements = sum(
            analysis.disagreement_count for analysis in consensus_results.values()
        )

        return {
            "overall_consensus_strength": overall_consensus,
            "total_disagreements": total_disagreements,
            "categories_with_claims": sum(1 for analysis in consensus_results.values() if analysis.total_claims > 0),
            "categories_with_disagreements": sum(
                1 for analysis in consensus_results.values() if analysis.disagreement_count > 0),
            "source_diversity_impact": source_analysis["diversity_score"],
            "authority_weighted_average": sum(
                analysis.authority_weighted_consensus * analysis.total_claims
                for analysis in consensus_results.values()
            ) / sum(analysis.total_claims for analysis in consensus_results.values()) if any(
                analysis.total_claims > 0 for analysis in consensus_results.values()) else 0.0
        }

    def _get_claim_distribution_with_models(self, all_claims: List[FactualClaim]) -> Dict[str, Any]:
        """Get distribution of claims across categories and sources using models"""

        category_distribution = Counter(claim.claim_category for claim in all_claims)
        source_distribution = Counter(claim.source_id for claim in all_claims)
        type_distribution = Counter(claim.claim_type for claim in all_claims)

        return {
            "by_category": dict(category_distribution),
            "by_source": dict(source_distribution),
            "by_type": dict(type_distribution),
            "total_claims": len(all_claims),
            "average_confidence": sum(claim.confidence for claim in all_claims) / len(all_claims) if all_claims else 0.0,
            "authority_distribution": Counter([claim.source_authority for claim in all_claims])
        }

    def _summarize_disagreements_with_models(self, consensus_results: Dict[str, ConsensusAnalysisResult]) -> List[Dict[str, Any]]:
        """Summarize the most significant disagreements using model data"""

        significant_disagreements = []

        for category, analysis in consensus_results.items():
            for disagreement in analysis.disagreements:
                # Calculate significance based on authority difference and value difference
                authority_diff = abs(disagreement["authority1"] - disagreement["authority2"])

                significant_disagreements.append({
                    "category": category,
                    "claim_type": disagreement["claim_type"],
                    "difference": disagreement["difference"],
                    "sources": [disagreement["source1"], disagreement["source2"]],
                    "authority_difference": authority_diff,
                    "significance": authority_diff * 0.5,  # Simple significance score
                    "claim_ids": [disagreement.get("claim1_id"), disagreement.get("claim2_id")]
                })

        # Sort by significance and return top disagreements
        significant_disagreements.sort(key=lambda x: x["significance"], reverse=True)
        return significant_disagreements[:5]  # Top 5 most significant

    # NOTE: The following methods remain UNCHANGED from the original implementation:
    # - _determine_claim_type()
    # - _values_agree()
    # - _calculate_difference()
    # - _extract_domain()
    # - _calculate_authority_variance()
    # - _create_unverifiable_result()
    # - _create_error_result()

    def _determine_claim_type(self, pattern: str, category: str) -> str:
        """Determine the specific type of claim based on pattern"""

        type_mapping = {
            "fuel_economy": {
                "mpg": "city_mpg" if "city" in pattern else "highway_mpg" if "highway" in pattern else "combined_mpg",
                "miles per gallon": "general_mpg",
                "fuel economy": "general_fuel_economy",
                "gas mileage": "general_fuel_economy"
            },
            "performance": {
                "hp": "horsepower",
                "horsepower": "horsepower",
                "lb-ft": "torque",
                "torque": "torque",
                "0-60": "acceleration_0_60",
                "quarter mile": "quarter_mile"
            },
            "pricing": {
                "$": "msrp",
                "price": "price",
                "cost": "cost",
                "msrp": "msrp"
            },
            "safety": {
                "star": "nhtsa_rating",
                "nhtsa": "nhtsa_rating",
                "iihs": "iihs_award",
                "safety rating": "general_safety"
            },
            "dimensions": {
                "long": "length",
                "wide": "width",
                "lbs": "weight",
                "pounds": "weight",
                "cargo": "cargo_volume"
            },
            "reliability": {
                "warranty": "warranty_years",
                "reliability": "reliability_score",
                "jd power": "jd_power_score",
                "problems": "problems_per_100"
            }
        }

        category_types = type_mapping.get(category, {})

        for key, claim_type in category_types.items():
            if key in pattern.lower():
                return claim_type

        return f"general_{category}"

    def _values_agree(self, val1: Any, val2: Any, claim_type: str) -> bool:
        """Check if two values agree within acceptable tolerance"""

        # Define tolerances for different claim types
        tolerances = {
            "horsepower": 0.05,  # 5% tolerance
            "torque": 0.05,  # 5% tolerance
            "city_mpg": 2,  # 2 MPG tolerance
            "highway_mpg": 2,  # 2 MPG tolerance
            "combined_mpg": 2,  # 2 MPG tolerance
            "weight": 0.03,  # 3% tolerance
            "length": 1,  # 1 inch tolerance
            "width": 1,  # 1 inch tolerance
            "msrp": 0.05,  # 5% tolerance for pricing
            "nhtsa_rating": 0,  # Exact match for ratings
            "acceleration_0_60": 0.2,  # 0.2 second tolerance
        }

        # Handle exact string matches
        if isinstance(val1, str) and isinstance(val2, str):
            return val1.lower().strip() == val2.lower().strip()

        # Handle numerical comparisons
        try:
            num1 = float(val1)
            num2 = float(val2)

            tolerance = tolerances.get(claim_type, 0.1)  # Default 10% tolerance

            if tolerance <= 1:  # Percentage tolerance
                max_val = max(num1, num2)
                if max_val == 0:
                    return num1 == num2
                return abs(num1 - num2) / max_val <= tolerance
            else:  # Absolute tolerance
                return abs(num1 - num2) <= tolerance

        except (ValueError, TypeError):
            # Fall back to string comparison
            return str(val1).lower().strip() == str(val2).lower().strip()

    def _calculate_difference(self, val1: Any, val2: Any) -> str:
        """Calculate the difference between two values"""

        try:
            num1 = float(val1)
            num2 = float(val2)

            diff = abs(num1 - num2)
            percent_diff = (diff / max(num1, num2)) * 100 if max(num1, num2) > 0 else 0

            return f"{diff:.1f} ({percent_diff:.1f}%)"

        except (ValueError, TypeError):
            return f"'{val1}' vs '{val2}'"

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""

        if not url:
            return ""

        import re
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        return match.group(1).lower() if match else ""

    def _calculate_authority_variance(self, authorities: List[float]) -> float:
        """Calculate variance in source authorities (higher variance = more diversity)"""

        if len(authorities) < 2:
            return 1.0

        mean_authority = sum(authorities) / len(authorities)
        variance = sum((auth - mean_authority) ** 2 for auth in authorities) / len(authorities)

        # Normalize variance (max possible variance is 0.25 for 0-1 range)
        normalized_variance = min(variance / 0.25, 1.0)

        return normalized_variance

    def _create_unverifiable_result(self, start_time: datetime,
                                    precondition_result: PreconditionResult) -> ValidationStepResult:
        """Create result for unverifiable validation"""

        return ValidationStepResult(
            step_id=f"consensus_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Consensus Analysis",
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
                    message="Cannot analyze consensus",
                    explanation=precondition_result.failure_reason,
                    suggestion="Ensure multiple diverse sources are available"
                )
            ]
        )

    def _create_error_result(self, start_time: datetime, error_message: str) -> ValidationStepResult:
        """Create result for validation errors"""

        return ValidationStepResult(
            step_id=f"consensus_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Consensus Analysis",
            status=ValidationStatus.FAILED,
            confidence_impact=-8.0,
            summary=f"Analysis failed: {error_message}",
            details={"error": error_message},
            started_at=start_time,
            completed_at=datetime.now(),
            warnings=[
                ValidationWarning(
                    category="validation_error",
                    severity="critical",
                    message="Consensus analysis encountered an error",
                    explanation=error_message,
                    suggestion="Check logs and retry analysis"
                )
            ]
        )