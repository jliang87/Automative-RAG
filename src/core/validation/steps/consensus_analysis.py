"""
Consensus Analysis Validator
Analyzes consensus across multiple sources for automotive information
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from collections import Counter, defaultdict

from ..models.validation_models import (
    ValidationStepResult, ValidationStatus, ValidationStepType,
    ValidationContext, ValidationWarning
)
from .steps_readiness_checker import MetaValidator, PreconditionResult

logger = logging.getLogger(__name__)


class ConsensusValidator:
    """
    Analyzes consensus across multiple sources for automotive information
    """

    def __init__(self, step_config: Dict[str, Any], meta_validator: MetaValidator):
        self.step_config = step_config
        self.meta_validator = meta_validator
        self.step_type = ValidationStepType.CONSENSUS

        # Define what constitutes factual claims for automotive content
        self.factual_claim_patterns = self._initialize_claim_patterns()

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
        """Perform the actual consensus analysis"""

        documents = context.documents
        warnings = []
        sources_analyzed = []

        # Extract claims from all documents
        all_claims = []
        for i, doc in enumerate(documents):
            doc_claims = self._extract_claims_from_document(doc, i)
            all_claims.extend(doc_claims)
            sources_analyzed.append(f"document_{i}")

        # Group claims by category and analyze consensus
        consensus_analysis = {
            "fuel_economy_consensus": self._analyze_category_consensus(all_claims, "fuel_economy"),
            "performance_consensus": self._analyze_category_consensus(all_claims, "performance"),
            "pricing_consensus": self._analyze_category_consensus(all_claims, "pricing"),
            "safety_consensus": self._analyze_category_consensus(all_claims, "safety"),
            "dimensions_consensus": self._analyze_category_consensus(all_claims, "dimensions"),
            "reliability_consensus": self._analyze_category_consensus(all_claims, "reliability")
        }

        # Analyze source diversity and reliability
        source_analysis = self._analyze_source_diversity(documents)

        # Calculate overall consensus metrics
        consensus_metrics = self._calculate_consensus_metrics(consensus_analysis, source_analysis)

        # Generate warnings for significant disagreements
        for category, analysis in consensus_analysis.items():
            if analysis["consensus_strength"] < 0.7 and analysis["claim_count"] >= 2:
                disagreements = analysis.get("disagreements", [])
                if disagreements:
                    severity = "critical" if analysis["consensus_strength"] < 0.5 else "caution"
                    warnings.append(ValidationWarning(
                        category=f"consensus_{category}",
                        severity=severity,
                        message=f"Sources disagree on {category.replace('_', ' ')}",
                        explanation=f"Consensus strength: {analysis['consensus_strength']:.1%}. Disagreements: {len(disagreements)}",
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
        total_claims = sum(analysis["claim_count"] for analysis in consensus_analysis.values())
        total_agreements = sum(analysis["agreement_count"] for analysis in consensus_analysis.values())

        summary = (f"Analyzed {total_claims} claims across {len(documents)} sources. "
                   f"Overall consensus: {overall_consensus:.1%}, "
                   f"Source diversity: {source_diversity:.1%}, "
                   f"Agreements: {total_agreements}/{total_claims}")

        # Build detailed results
        details = {
            "sources_analyzed": len(documents),
            "total_claims_found": total_claims,
            "total_agreements": total_agreements,
            "overall_consensus_strength": overall_consensus,
            "source_diversity_score": source_diversity,
            "consensus_by_category": consensus_analysis,
            "source_analysis": source_analysis,
            "consensus_metrics": consensus_metrics,
            "claim_distribution": self._get_claim_distribution(all_claims),
            "disagreement_summary": self._summarize_disagreements(consensus_analysis)
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

    def _extract_claims_from_document(self, doc: Dict[str, Any], doc_index: int) -> List[Dict[str, Any]]:
        """Extract factual claims from a document"""

        content = doc.get("content", "")
        metadata = doc.get("metadata", {})

        claims = []
        source_id = metadata.get("url", f"document_{doc_index}")
        source_type = self._determine_source_type(metadata)
        source_authority = self._assess_source_authority(metadata)

        # Extract claims for each category
        for category, patterns in self.factual_claim_patterns.items():
            category_claims = self._extract_category_claims(content, category, patterns)
            for claim in category_claims:
                claims.append({
                    "category": category,
                    "claim_type": claim["type"],
                    "value": claim["value"],
                    "raw_text": claim["raw_text"],
                    "source_id": source_id,
                    "source_type": source_type,
                    "source_authority": source_authority,
                    "document_index": doc_index
                })

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
                    "pattern_used": pattern
                })

        return claims

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

    def _analyze_category_consensus(self, all_claims: List[Dict], category: str) -> Dict[str, Any]:
        """Analyze consensus for a specific category of claims"""

        # Filter claims for this category
        category_claims = [claim for claim in all_claims if claim["category"] == category]

        if len(category_claims) < 2:
            return {
                "consensus_strength": 1.0,  # No disagreement if only one claim
                "claim_count": len(category_claims),
                "agreement_count": len(category_claims),
                "disagreements": [],
                "claim_groups": {},
                "analysis": "insufficient_data"
            }

        # Group claims by type
        claims_by_type = defaultdict(list)
        for claim in category_claims:
            claims_by_type[claim["claim_type"]].append(claim)

        agreements = 0
        total_comparisons = 0
        disagreements = []
        claim_groups = {}

        # Analyze each claim type
        for claim_type, claims in claims_by_type.items():
            if len(claims) < 2:
                continue

            type_analysis = self._analyze_claim_type_consensus(claims, claim_type)
            claim_groups[claim_type] = type_analysis

            agreements += type_analysis["agreements"]
            total_comparisons += type_analysis["total_comparisons"]
            disagreements.extend(type_analysis["disagreements"])

        # Calculate consensus strength
        if total_comparisons > 0:
            consensus_strength = agreements / total_comparisons
        else:
            consensus_strength = 1.0

        return {
            "consensus_strength": consensus_strength,
            "claim_count": len(category_claims),
            "agreement_count": agreements,
            "total_comparisons": total_comparisons,
            "disagreements": disagreements,
            "claim_groups": claim_groups,
            "analysis": "complete" if total_comparisons > 0 else "insufficient_data"
        }

    def _analyze_claim_type_consensus(self, claims: List[Dict], claim_type: str) -> Dict[str, Any]:
        """Analyze consensus for a specific type of claim"""

        if len(claims) < 2:
            return {
                "agreements": 0,
                "total_comparisons": 0,
                "disagreements": [],
                "values": [claim["value"] for claim in claims],
                "consensus_value": claims[0]["value"] if claims else None
            }

        # Extract values and sources
        values = []
        for claim in claims:
            values.append({
                "value": claim["value"],
                "source_id": claim["source_id"],
                "source_authority": claim["source_authority"],
                "raw_text": claim["raw_text"]
            })

        # Check for agreements and disagreements
        agreements = 0
        total_comparisons = 0
        disagreements = []

        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                total_comparisons += 1

                val1, val2 = values[i]["value"], values[j]["value"]

                if self._values_agree(val1, val2, claim_type):
                    agreements += 1
                else:
                    disagreements.append({
                        "claim_type": claim_type,
                        "value1": val1,
                        "value2": val2,
                        "source1": values[i]["source_id"],
                        "source2": values[j]["source_id"],
                        "authority1": values[i]["source_authority"],
                        "authority2": values[j]["source_authority"],
                        "difference": self._calculate_difference(val1, val2)
                    })

        # Determine consensus value (most authoritative or most common)
        consensus_value = self._determine_consensus_value(values, claim_type)

        return {
            "agreements": agreements,
            "total_comparisons": total_comparisons,
            "disagreements": disagreements,
            "values": values,
            "consensus_value": consensus_value,
            "value_distribution": Counter([v["value"] for v in values])
        }

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

    def _determine_consensus_value(self, values: List[Dict], claim_type: str) -> Any:
        """Determine the consensus value from multiple claims"""

        if not values:
            return None

        # Weight by source authority
        weighted_values = []
        for value_info in values:
            weight = value_info["source_authority"]
            weighted_values.append((value_info["value"], weight))

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
            max_authority = max(value_info["source_authority"] for value_info in values)
            authoritative_values = [
                value_info["value"] for value_info in values
                if value_info["source_authority"] == max_authority
            ]

            # Return most common among most authoritative
            return Counter(authoritative_values).most_common(1)[0][0] if authoritative_values else None

    def _analyze_source_diversity(self, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze diversity of sources"""

        source_types = []
        source_authorities = []
        source_domains = set()

        for doc in documents:
            metadata = doc.get("metadata", {})

            source_type = self._determine_source_type(metadata)
            source_authority = self._assess_source_authority(metadata)
            source_domain = self._extract_domain(metadata.get("url", ""))

            source_types.append(source_type)
            source_authorities.append(source_authority)
            if source_domain:
                source_domains.add(source_domain)

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
            "source_type_distribution": Counter(source_types),
            "unique_domains": len(source_domains),
            "authority_range": {
                "min": min(source_authorities) if source_authorities else 0,
                "max": max(source_authorities) if source_authorities else 0,
                "avg": sum(source_authorities) / len(source_authorities) if source_authorities else 0
            }
        }

    def _determine_source_type(self, metadata: Dict[str, Any]) -> str:
        """Determine the type of source"""

        url = metadata.get("url", "").lower()

        if any(domain in url for domain in ['.gov', 'epa.', 'nhtsa.']):
            return "regulatory"
        elif any(domain in url for domain in ['toyota.com', 'honda.com', 'ford.com', 'bmw.com']):
            return "official"
        elif any(domain in url for domain in ['edmunds.com', 'motortrend.com', 'caranddriver.com']):
            return "professional"
        elif any(domain in url for domain in ['reddit.com', 'forum', 'review']):
            return "user_generated"
        elif any(domain in url for domain in ['.edu', 'sae.org']):
            return "academic"
        else:
            return "other"

    def _assess_source_authority(self, metadata: Dict[str, Any]) -> float:
        """Assess the authority score of a source"""

        source_type = self._determine_source_type(metadata)

        authority_scores = {
            "regulatory": 1.0,
            "official": 0.9,
            "academic": 0.85,
            "professional": 0.75,
            "user_generated": 0.4,
            "other": 0.5
        }

        return authority_scores.get(source_type, 0.5)

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

    def _calculate_consensus_metrics(self, consensus_analysis: Dict[str, Any], source_analysis: Dict[str, Any]) -> Dict[
        str, Any]:
        """Calculate overall consensus metrics"""

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

        for category, analysis in consensus_analysis.items():
            if analysis["claim_count"] > 0:  # Only include categories with claims
                weight = category_weights.get(category, 0.1)
                weighted_consensus += analysis["consensus_strength"] * weight
                total_weight += weight

        overall_consensus = weighted_consensus / total_weight if total_weight > 0 else 1.0

        # Count total disagreements
        total_disagreements = sum(
            len(analysis.get("disagreements", []))
            for analysis in consensus_analysis.values()
        )

        return {
            "overall_consensus_strength": overall_consensus,
            "total_disagreements": total_disagreements,
            "categories_with_claims": sum(1 for analysis in consensus_analysis.values() if analysis["claim_count"] > 0),
            "categories_with_disagreements": sum(
                1 for analysis in consensus_analysis.values() if analysis.get("disagreements")),
            "source_diversity_impact": source_analysis["diversity_score"]
        }

    def _get_claim_distribution(self, all_claims: List[Dict]) -> Dict[str, Any]:
        """Get distribution of claims across categories and sources"""

        category_distribution = Counter(claim["category"] for claim in all_claims)
        source_distribution = Counter(claim["source_id"] for claim in all_claims)

        return {
            "by_category": dict(category_distribution),
            "by_source": dict(source_distribution),
            "total_claims": len(all_claims)
        }

    def _summarize_disagreements(self, consensus_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Summarize the most significant disagreements"""

        significant_disagreements = []

        for category, analysis in consensus_analysis.items():
            disagreements = analysis.get("disagreements", [])

            for disagreement in disagreements:
                # Calculate significance based on authority difference and value difference
                authority_diff = abs(disagreement["authority1"] - disagreement["authority2"])

                significant_disagreements.append({
                    "category": category,
                    "claim_type": disagreement["claim_type"],
                    "difference": disagreement["difference"],
                    "sources": [disagreement["source1"], disagreement["source2"]],
                    "authority_difference": authority_diff,
                    "significance": authority_diff * 0.5  # Simple significance score
                })

        # Sort by significance and return top disagreements
        significant_disagreements.sort(key=lambda x: x["significance"], reverse=True)
        return significant_disagreements[:5]  # Top 5 most significant

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