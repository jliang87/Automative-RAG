"""
Confidence Calculator
Multi-dimensional confidence scoring with meta-validation adjustments
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from src.models import (
    ValidationStepResult, ValidationStepType, ValidationStatus,
    ConfidenceBreakdown, ConfidenceLevel
)
from src.models.knowledge_models import (
    ConfidenceWeights, ConfidenceCalculationConfig,
    SourceAuthority, ValidationReferenceDatabase
)

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """
    Calculates multi-dimensional confidence scores using model-driven configuration
    """

    def __init__(self,
                 config: Optional[ConfidenceCalculationConfig] = None,
                 source_authorities: Optional[List[SourceAuthority]] = None):
        # Use model-based configuration instead of hardcoded weights
        self.config = config or ConfidenceCalculationConfig()

        # Build authority lookup from model data
        self.source_authority_lookup = {}
        if source_authorities:
            for authority in source_authorities:
                self.source_authority_lookup[authority.domain] = authority

        # Legacy compatibility - convert to dict for existing code
        self.default_weights = {
            ValidationStepType.SOURCE_CREDIBILITY: self.config.weights.source_credibility,
            ValidationStepType.TECHNICAL_CONSISTENCY: self.config.weights.technical_consistency,
            ValidationStepType.COMPLETENESS: self.config.weights.completeness,
            ValidationStepType.CONSENSUS: self.config.weights.consensus,
            ValidationStepType.LLM_INFERENCE: self.config.weights.llm_inference,
            ValidationStepType.RETRIEVAL: self.config.weights.retrieval
        }

        # Status score multipliers remain the same
        self.status_multipliers = {
            ValidationStatus.PASSED: 1.0,
            ValidationStatus.WARNING: 0.7,
            ValidationStatus.FAILED: 0.0,
            ValidationStatus.UNVERIFIABLE: 0.0,
            ValidationStatus.PENDING: 0.0
        }

    def get_source_authority_score(self, domain: str) -> float:
        """Get authority score from model data instead of hardcoded lookup"""
        if domain in self.source_authority_lookup:
            return self.source_authority_lookup[domain].authority_score
        return 0.5  # Default for unknown sources

    def calculate_confidence(
        self,
        step_results: List[ValidationStepResult],
        custom_weights: Optional[Dict[ValidationStepType, float]] = None
    ) -> ConfidenceBreakdown:
        """
        Calculate comprehensive confidence score with meta-validation adjustments
        """

        if not step_results:
            return ConfidenceBreakdown(
                total_score=0.0,
                level=ConfidenceLevel.POOR,
                verification_coverage=0.0
            )

        # Use custom weights if provided, otherwise use defaults
        weights = custom_weights or self.default_weights

        # Calculate component scores
        component_scores = self._calculate_component_scores(step_results, weights)

        # Calculate verification coverage (what percentage of steps could be verified)
        verification_coverage = self._calculate_verification_coverage(step_results, weights)

        # Calculate base confidence from verifiable steps
        base_confidence = self._calculate_base_confidence(step_results, weights)

        # Apply meta-validation adjustments
        adjusted_confidence = self._apply_meta_validation_adjustments(
            base_confidence, verification_coverage, step_results
        )

        # Determine confidence level
        confidence_level = self._determine_confidence_level(adjusted_confidence)

        # Create detailed breakdown
        breakdown = ConfidenceBreakdown(
            total_score=adjusted_confidence,
            level=confidence_level,
            source_credibility=component_scores.get(ValidationStepType.SOURCE_CREDIBILITY, 0.0),
            technical_consistency=component_scores.get(ValidationStepType.TECHNICAL_CONSISTENCY, 0.0),
            completeness=component_scores.get(ValidationStepType.COMPLETENESS, 0.0),
            consensus=component_scores.get(ValidationStepType.CONSENSUS, 0.0),
            llm_quality=component_scores.get(ValidationStepType.LLM_INFERENCE, 0.0),
            verification_coverage=verification_coverage,
            unverifiable_penalty=max(0, 100 - verification_coverage),
            calculation_method="weighted_average_with_meta_validation",
            weights_used=weights
        )

        logger.info(f"Calculated confidence: {adjusted_confidence:.1f}% ({confidence_level.value})")
        logger.info(f"Verification coverage: {verification_coverage:.1f}%")

        return breakdown

    def _calculate_component_scores(
        self,
        step_results: List[ValidationStepResult],
        weights: Dict[ValidationStepType, float]
    ) -> Dict[ValidationStepType, float]:
        """Calculate individual component confidence scores"""

        component_scores = {}

        for step_result in step_results:
            step_type = step_result.step_type

            # Skip unverifiable steps for component scoring
            if step_result.status == ValidationStatus.UNVERIFIABLE:
                continue

            # Calculate base score from status
            status_multiplier = self.status_multipliers.get(step_result.status, 0.0)

            # Add confidence impact from step
            confidence_impact = step_result.confidence_impact

            # Combine status and impact (base score + impact bonus)
            base_score = 80.0  # Base score for passed validations
            impact_bonus = confidence_impact

            step_score = (base_score * status_multiplier) + impact_bonus

            # Apply any step-specific adjustments
            step_score = self._apply_step_specific_adjustments(step_result, step_score)

            # Cap at 100
            step_score = min(100.0, max(0.0, step_score))

            component_scores[step_type] = step_score

            logger.debug(
                f"Component score for {step_type.value}: {step_score:.1f} "
                f"(status: {step_result.status}, impact: {confidence_impact})"
            )

        return component_scores

    def _calculate_verification_coverage(
        self,
        step_results: List[ValidationStepResult],
        weights: Dict[ValidationStepType, float]
    ) -> float:
        """Calculate what percentage of validation could be performed"""

        total_weight = 0.0
        verifiable_weight = 0.0

        for step_result in step_results:
            step_weight = weights.get(step_result.step_type, 0.0)
            total_weight += step_weight

            if step_result.status != ValidationStatus.UNVERIFIABLE:
                verifiable_weight += step_weight

        if total_weight == 0:
            return 0.0

        coverage = (verifiable_weight / total_weight) * 100
        return min(100.0, max(0.0, coverage))

    def _calculate_base_confidence(
        self,
        step_results: List[ValidationStepResult],
        weights: Dict[ValidationStepType, float]
    ) -> float:
        """Calculate base confidence from verifiable steps only"""

        weighted_scores = []
        total_verifiable_weight = 0.0

        for step_result in step_results:
            if step_result.status == ValidationStatus.UNVERIFIABLE:
                continue

            step_weight = weights.get(step_result.step_type, 0.0)
            if step_weight == 0:
                continue

            # Calculate step confidence score
            status_multiplier = self.status_multipliers.get(step_result.status, 0.0)
            base_score = 80.0  # Base score for successful validation
            impact_bonus = step_result.confidence_impact

            step_score = (base_score * status_multiplier) + impact_bonus
            step_score = min(100.0, max(0.0, step_score))

            # Weight the score
            weighted_score = step_score * step_weight
            weighted_scores.append(weighted_score)
            total_verifiable_weight += step_weight

        if total_verifiable_weight == 0:
            return 0.0

        # Calculate weighted average of verifiable steps
        total_weighted_score = sum(weighted_scores)
        base_confidence = total_weighted_score / total_verifiable_weight

        return min(100.0, max(0.0, base_confidence))

    def _apply_meta_validation_adjustments(
        self,
        base_confidence: float,
        verification_coverage: float,
        step_results: List[ValidationStepResult]
    ) -> float:
        """Apply meta-validation adjustments to base confidence"""

        # Start with base confidence calculated from verifiable steps
        adjusted_confidence = base_confidence

        # Apply verification coverage penalty
        coverage_penalty = self._calculate_coverage_penalty(verification_coverage)
        adjusted_confidence *= (1.0 - coverage_penalty)

        # Apply quality adjustments based on validation depth
        quality_multiplier = self._calculate_quality_multiplier(step_results)
        adjusted_confidence *= quality_multiplier

        # Apply warning penalties
        warning_penalty = self._calculate_warning_penalty(step_results)
        adjusted_confidence *= (1.0 - warning_penalty)

        # Apply failure penalties
        failure_penalty = self._calculate_failure_penalty(step_results)
        adjusted_confidence *= (1.0 - failure_penalty)

        return min(100.0, max(0.0, adjusted_confidence))

    def _calculate_coverage_penalty(self, verification_coverage: float) -> float:
        """Calculate penalty for incomplete verification coverage"""

        if verification_coverage >= 90:
            return 0.0  # No penalty for high coverage
        elif verification_coverage >= 70:
            return 0.1  # Small penalty for good coverage
        elif verification_coverage >= 50:
            return 0.2  # Medium penalty for partial coverage
        else:
            return 0.3  # Large penalty for poor coverage

    def _calculate_quality_multiplier(self, step_results: List[ValidationStepResult]) -> float:
        """Calculate quality multiplier based on validation depth"""

        quality_factors = []

        for step_result in step_results:
            if step_result.status == ValidationStatus.UNVERIFIABLE:
                continue

            # Factor 1: Number of sources used
            sources_count = len(step_result.sources_used)
            if sources_count >= 3:
                quality_factors.append(1.1)  # Boost for multiple sources
            elif sources_count >= 2:
                quality_factors.append(1.05)  # Small boost for dual sources
            else:
                quality_factors.append(1.0)  # No penalty for single source

            # Factor 2: Detailed validation (presence of detailed results)
            if step_result.details and len(step_result.details) > 2:
                quality_factors.append(1.05)  # Boost for detailed validation

            # Factor 3: Step-specific quality indicators
            if step_result.step_type == ValidationStepType.TECHNICAL_CONSISTENCY:
                # Check for physics/engineering validation depth
                if any(key in step_result.details for key in ['physics_check', 'engineering_validation', 'constraint_analysis']):
                    quality_factors.append(1.1)

            if step_result.step_type == ValidationStepType.SOURCE_CREDIBILITY:
                # Check for authority scoring depth
                if 'authority_scores' in step_result.details:
                    quality_factors.append(1.05)

        if not quality_factors:
            return 1.0

        # Calculate average quality multiplier
        avg_multiplier = statistics.mean(quality_factors)
        return min(1.2, max(0.8, avg_multiplier))  # Cap between 0.8 and 1.2

    def _calculate_warning_penalty(self, step_results: List[ValidationStepResult]) -> float:
        """Calculate penalty for validation warnings"""

        warning_count = 0
        total_warnings = 0

        for step_result in step_results:
            if step_result.status == ValidationStatus.WARNING:
                warning_count += 1
            total_warnings += len(step_result.warnings)

        if warning_count == 0 and total_warnings == 0:
            return 0.0

        # Base penalty for warning status
        status_penalty = warning_count * 0.05  # 5% per warning status

        # Additional penalty for multiple specific warnings
        warning_penalty = min(total_warnings * 0.02, 0.1)  # 2% per warning, max 10%

        return min(0.2, status_penalty + warning_penalty)  # Max 20% penalty

    def _calculate_failure_penalty(self, step_results: List[ValidationStepResult]) -> float:
        """Calculate penalty for validation failures"""

        failure_count = sum(1 for result in step_results if result.status == ValidationStatus.FAILED)
        total_steps = len(step_results)

        if failure_count == 0:
            return 0.0

        # Heavy penalty for failures
        failure_rate = failure_count / total_steps

        if failure_rate >= 0.5:
            return 0.5  # 50% penalty for high failure rate
        elif failure_rate >= 0.25:
            return 0.3  # 30% penalty for moderate failure rate
        else:
            return failure_rate * 0.4  # 40% of failure rate as penalty

    def _apply_step_specific_adjustments(
        self,
        step_result: ValidationStepResult,
        base_score: float
    ) -> float:
        """Apply step-specific score adjustments"""

        adjusted_score = base_score

        # Technical consistency specific adjustments
        if step_result.step_type == ValidationStepType.TECHNICAL_CONSISTENCY:
            # Boost for physics validation
            if 'physics_validated' in step_result.details and step_result.details['physics_validated']:
                adjusted_score += 5.0

            # Boost for official source verification
            if 'official_sources_used' in step_result.details:
                official_count = step_result.details['official_sources_used']
                adjusted_score += min(10.0, official_count * 3.0)

        # Source credibility specific adjustments
        elif step_result.step_type == ValidationStepType.SOURCE_CREDIBILITY:
            # Boost for high authority sources
            if 'avg_authority_score' in step_result.details:
                avg_authority = step_result.details['avg_authority_score']
                if avg_authority >= 0.8:
                    adjusted_score += 10.0
                elif avg_authority >= 0.6:
                    adjusted_score += 5.0

            # Penalty for bias detection
            if 'bias_detected' in step_result.details and step_result.details['bias_detected']:
                adjusted_score -= 10.0

        # Consensus specific adjustments
        elif step_result.step_type == ValidationStepType.CONSENSUS:
            # Boost for strong consensus
            if 'consensus_strength' in step_result.details:
                consensus = step_result.details['consensus_strength']
                if consensus >= 0.8:
                    adjusted_score += 8.0
                elif consensus >= 0.6:
                    adjusted_score += 4.0

            # Boost for source diversity
            if 'source_type_diversity' in step_result.details:
                diversity = step_result.details['source_type_diversity']
                adjusted_score += min(5.0, diversity * 2.0)

        return adjusted_score

    def _determine_confidence_level(self, confidence_score: float) -> ConfidenceLevel:
        """Determine confidence level from numeric score"""

        if confidence_score >= 90:
            return ConfidenceLevel.EXCELLENT
        elif confidence_score >= 80:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 70:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 60:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.POOR

    def calculate_confidence_improvement(
        self,
        original_result: ConfidenceBreakdown,
        updated_steps: List[ValidationStepResult],
        weights: Dict[ValidationStepType, float]
    ) -> float:
        """Calculate confidence improvement after step updates"""

        new_result = self.calculate_confidence(updated_steps, weights)
        improvement = new_result.total_score - original_result.total_score

        logger.info(f"Confidence improvement: {improvement:.1f}% ({original_result.total_score:.1f}% → {new_result.total_score:.1f}%)")

        return improvement

    def get_confidence_explanation(self, breakdown: ConfidenceBreakdown) -> Dict[str, str]:
        """Generate human-readable explanations for confidence scores"""

        explanations = {}

        # Overall confidence explanation
        if breakdown.level == ConfidenceLevel.EXCELLENT:
            explanations['overall'] = "Excellent validation quality with authoritative sources and comprehensive verification."
        elif breakdown.level == ConfidenceLevel.HIGH:
            explanations['overall'] = "High confidence with good source quality and successful validation checks."
        elif breakdown.level == ConfidenceLevel.MEDIUM:
            explanations['overall'] = "Moderate confidence with some limitations in sources or validation coverage."
        elif breakdown.level == ConfidenceLevel.LOW:
            explanations['overall'] = "Limited confidence due to validation gaps or source quality issues."
        else:
            explanations['overall'] = "Low confidence due to significant validation limitations or failures."

        # Coverage explanation
        if breakdown.verification_coverage >= 90:
            explanations['coverage'] = "Comprehensive validation coverage with all major checks completed."
        elif breakdown.verification_coverage >= 70:
            explanations['coverage'] = "Good validation coverage with most checks successfully completed."
        elif breakdown.verification_coverage >= 50:
            explanations['coverage'] = "Partial validation coverage with some checks unable to complete."
        else:
            explanations['coverage'] = "Limited validation coverage due to missing reference data or system limitations."

        # Component explanations
        if breakdown.source_credibility >= 80:
            explanations['sources'] = "High-quality, authoritative sources with good reliability scores."
        elif breakdown.source_credibility >= 60:
            explanations['sources'] = "Reasonable source quality with some limitations in authority or bias."
        else:
            explanations['sources'] = "Limited source quality or credibility assessment challenges."

        if breakdown.technical_consistency >= 80:
            explanations['technical'] = "Strong technical validation with official references and constraint checking."
        elif breakdown.technical_consistency >= 60:
            explanations['technical'] = "Adequate technical validation with some reference limitations."
        else:
            explanations['technical'] = "Limited technical validation due to missing reference data."

        return explanations