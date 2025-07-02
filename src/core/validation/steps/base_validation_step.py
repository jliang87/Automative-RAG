"""
Base class for all validation steps
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional

from ..models.validation_models import (
    ValidationStepResult, ValidationStepType, ValidationContext,
    ValidationStatus, ValidationStepConfig, ValidationWarning,
    create_validation_step_result
)
from ..meta_validator import MetaValidator, PreconditionResult

logger = logging.getLogger(__name__)


class BaseValidationStep(ABC):
    """
    Abstract base class for validation steps with meta-validation support
    """

    def __init__(self, config: ValidationStepConfig, meta_validator: MetaValidator):
        self.config = config
        self.meta_validator = meta_validator
        self.step_type = config.step_type
        self.step_name = config.step_name

    async def execute(self, context: ValidationContext) -> ValidationStepResult:
        """
        Execute validation step with meta-validation (validation of the validation)
        """

        start_time = datetime.now()

        logger.info(f"Starting validation step: {self.step_name}")

        # Step 1: Meta-validation - check if this step can execute properly
        precondition_result = await self.meta_validator.check_preconditions(
            self.step_type, context, self.config.parameters
        )

        # Step 2: Handle precondition failures
        if precondition_result.status == "UNVERIFIABLE":
            return self._create_unverifiable_result(start_time, precondition_result, context)

        # Step 3: Execute main validation logic
        try:
            # Execute the actual validation
            result = await self._execute_validation(context, precondition_result)

            # Set timing information
            result.started_at = start_time
            result.completed_at = datetime.now()
            if result.completed_at:
                duration = result.completed_at - start_time
                result.duration_ms = int(duration.total_seconds() * 1000)

            # Add any precondition warnings
            if precondition_result.status == "PARTIAL":
                for failure in precondition_result.missing_resources:
                    warning = ValidationWarning(
                        category="precondition_limitation",
                        severity="caution",
                        message=failure.failure_reason,
                        explanation=failure.impact_description,
                        suggestion=failure.suggested_action
                    )
                    result.warnings.append(warning)

            logger.info(f"Validation step {self.step_name} completed with status: {result.status}")
            return result

        except Exception as e:
            logger.error(f"Error in validation step {self.step_name}: {str(e)}")
            return self._create_error_result(start_time, str(e))

    def _create_unverifiable_result(
            self,
            start_time: datetime,
            precondition_result: PreconditionResult,
            context: ValidationContext
    ) -> ValidationStepResult:
        """Create result for unverifiable validation step"""

        # Generate guidance for user
        guidance = self.meta_validator.guidance_generator.generate_guidance(
            self.step_type,
            precondition_result.missing_resources,
            context
        )

        # Create contribution prompt
        contribution_prompt = self.meta_validator.guidance_generator.create_contribution_prompt(
            guidance,
            self.step_type
        )

        result = create_validation_step_result(
            step_type=self.step_type,
            status=ValidationStatus.UNVERIFIABLE,
            summary=precondition_result.failure_reason,
            confidence_impact=0.0
        )

        result.step_name = self.step_name
        result.started_at = start_time
        result.completed_at = datetime.now()
        result.precondition_failures = precondition_result.missing_resources
        result.guidance = guidance
        result.contribution_prompt = contribution_prompt
        result.auto_retry_enabled = True

        # Add details about what was missing
        result.details = {
            "unverifiable_reason": precondition_result.failure_reason,
            "missing_resources": [
                {
                    "type": failure.resource_type,
                    "name": failure.resource_name,
                    "reason": failure.failure_reason,
                    "impact": failure.impact_description,
                    "suggestion": failure.suggested_action
                }
                for failure in precondition_result.missing_resources
            ],
            "guidance_available": True,
            "contribution_opportunity": True
        }

        return result

    def _create_error_result(self, start_time: datetime, error_message: str) -> ValidationStepResult:
        """Create result for validation step that encountered an error"""

        result = create_validation_step_result(
            step_type=self.step_type,
            status=ValidationStatus.FAILED,
            summary=f"Validation step failed: {error_message}",
            confidence_impact=0.0
        )

        result.step_name = self.step_name
        result.started_at = start_time
        result.completed_at = datetime.now()
        result.details = {
            "error": error_message,
            "error_type": "execution_error"
        }

        return result

    @abstractmethod
    async def _execute_validation(
            self,
            context: ValidationContext,
            precondition_result: PreconditionResult
    ) -> ValidationStepResult:
        """
        Execute the main validation logic for this step
        Must be implemented by subclasses
        """
        pass

    def _add_warning(
            self,
            result: ValidationStepResult,
            category: str,
            severity: str,
            message: str,
            explanation: str,
            suggestion: Optional[str] = None
    ):
        """Helper method to add warnings to validation results"""

        warning = ValidationWarning(
            category=category,
            severity=severity,
            message=message,
            explanation=explanation,
            suggestion=suggestion
        )
        result.warnings.append(warning)

    def _calculate_step_confidence_impact(
            self,
            base_score: float,
            quality_factors: Dict[str, float]
    ) -> float:
        """Calculate confidence impact for this step"""

        # Start with base score
        impact = base_score

        # Apply quality factors
        for factor_name, factor_value in quality_factors.items():
            if factor_name == "source_authority":
                impact += factor_value * 10  # High impact for authority
            elif factor_name == "source_diversity":
                impact += factor_value * 5  # Medium impact for diversity
            elif factor_name == "data_freshness":
                impact += factor_value * 3  # Lower impact for freshness
            elif factor_name == "consensus_strength":
                impact += factor_value * 8  # High impact for consensus
            elif factor_name == "technical_accuracy":
                impact += factor_value * 12  # Very high impact for technical accuracy

        # Cap the impact
        return min(30.0, max(-10.0, impact))

    def _extract_vehicle_context(self, context: ValidationContext) -> Dict[str, Any]:
        """Extract vehicle-specific context information"""

        vehicle_context = {}

        # Extract from explicit context fields
        if context.manufacturer:
            vehicle_context['manufacturer'] = context.manufacturer.lower()
        if context.model:
            vehicle_context['model'] = context.model
        if context.year:
            vehicle_context['year'] = context.year
        if context.trim:
            vehicle_context['trim'] = context.trim

        # Extract from documents if not in context
        if not vehicle_context and context.documents:
            for doc in context.documents:
                metadata = doc.get('metadata', {})

                # Try to extract vehicle info from metadata
                for field in ['manufacturer', 'vehicleModel', 'model', 'modelYear', 'year']:
                    if field in metadata and metadata[field]:
                        if field in ['manufacturer']:
                            vehicle_context['manufacturer'] = str(metadata[field]).lower()
                        elif field in ['vehicleModel', 'model']:
                            vehicle_context['model'] = str(metadata[field])
                        elif field in ['modelYear', 'year']:
                            try:
                                vehicle_context['year'] = int(metadata[field])
                            except (ValueError, TypeError):
                                pass

                # Stop if we found vehicle info
                if vehicle_context:
                    break

        return vehicle_context

    def _assess_source_quality(self, documents: list) -> Dict[str, float]:
        """Assess overall source quality metrics"""

        if not documents:
            return {"authority": 0.0, "diversity": 0.0, "freshness": 0.0}

        authority_scores = []
        source_types = set()
        freshness_scores = []

        for doc in documents:
            metadata = doc.get('metadata', {})

            # Authority assessment
            source_type = metadata.get('source_type', metadata.get('source', 'unknown'))
            if source_type in ['official', 'manufacturer', 'epa', 'nhtsa']:
                authority_scores.append(1.0)
            elif source_type in ['professional', 'automotive_journalism']:
                authority_scores.append(0.8)
            elif source_type in ['user_generated', 'forum', 'review']:
                authority_scores.append(0.4)
            else:
                authority_scores.append(0.2)

            source_types.add(source_type)

            # Freshness assessment (if publication date available)
            pub_date = metadata.get('publishedDate', metadata.get('upload_date'))
            if pub_date:
                try:
                    if isinstance(pub_date, str) and len(pub_date) >= 4:
                        year = int(pub_date[:4])
                        current_year = datetime.now().year
                        age = current_year - year
                        freshness = max(0.0, 1.0 - (age * 0.1))  # Decay 10% per year
                        freshness_scores.append(freshness)
                except (ValueError, TypeError):
                    pass

        # Calculate averages
        avg_authority = sum(authority_scores) / len(authority_scores) if authority_scores else 0.0
        diversity = min(1.0, len(source_types) / 3.0)  # Full diversity at 3+ source types
        avg_freshness = sum(freshness_scores) / len(freshness_scores) if freshness_scores else 0.5

        return {
            "authority": avg_authority,
            "diversity": diversity,
            "freshness": avg_freshness
        }