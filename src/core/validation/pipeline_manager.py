"""
Validation Pipeline Manager
Manages scenario-specific validation pipelines and orchestrates validation execution
"""

import logging
from typing import Dict, List, Optional, Type
from datetime import datetime

from .models.validation_models import (
    PipelineType, ValidationStepType, ValidationPipelineConfig,
    ValidationContext, ValidationChainResult, ValidationStepResult,
    ValidationStatus, ConfidenceBreakdown, ConfidenceLevel,
    create_pipeline_config
)
from .meta_validator import MetaValidator
from .confidence_calculator import ConfidenceCalculator

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    Manages validation pipelines for different query scenarios
    """

    def __init__(self):
        self.meta_validator = MetaValidator()
        self.confidence_calculator = ConfidenceCalculator()
        self.pipeline_configs = self._initialize_pipeline_configs()
        self.validation_steps = self._register_validation_steps()

    def _initialize_pipeline_configs(self) -> Dict[PipelineType, ValidationPipelineConfig]:
        """Initialize all scenario-specific pipeline configurations"""

        configs = {}

        # Pipeline 1: Specification Verification (Facts Query)
        configs[PipelineType.SPECIFICATION_VERIFICATION] = create_pipeline_config(
            pipeline_type=PipelineType.SPECIFICATION_VERIFICATION,
            steps=[
                ValidationStepType.RETRIEVAL,
                ValidationStepType.SOURCE_CREDIBILITY,
                ValidationStepType.TECHNICAL_CONSISTENCY,
                ValidationStepType.COMPLETENESS,
                ValidationStepType.LLM_INFERENCE
            ],
            confidence_weights={
                ValidationStepType.SOURCE_CREDIBILITY: 0.3,
                ValidationStepType.TECHNICAL_CONSISTENCY: 0.4,
                ValidationStepType.COMPLETENESS: 0.2,
                ValidationStepType.LLM_INFERENCE: 0.1
            }
        )

        # Pipeline 2: Feature Comparison (Features Query)
        configs[PipelineType.FEATURE_COMPARISON] = create_pipeline_config(
            pipeline_type=PipelineType.FEATURE_COMPARISON,
            steps=[
                ValidationStepType.RETRIEVAL,
                ValidationStepType.SOURCE_CREDIBILITY,
                ValidationStepType.CONSENSUS,
                ValidationStepType.COMPLETENESS,
                ValidationStepType.LLM_INFERENCE
            ],
            confidence_weights={
                ValidationStepType.SOURCE_CREDIBILITY: 0.3,
                ValidationStepType.CONSENSUS: 0.4,
                ValidationStepType.COMPLETENESS: 0.2,
                ValidationStepType.LLM_INFERENCE: 0.1
            }
        )

        # Pipeline 3: Trade-off Analysis (Tradeoffs Query)
        configs[PipelineType.TRADEOFF_ANALYSIS] = create_pipeline_config(
            pipeline_type=PipelineType.TRADEOFF_ANALYSIS,
            steps=[
                ValidationStepType.RETRIEVAL,
                ValidationStepType.SOURCE_CREDIBILITY,
                ValidationStepType.CONSENSUS,
                ValidationStepType.COMPLETENESS,
                ValidationStepType.LLM_INFERENCE
            ],
            confidence_weights={
                ValidationStepType.SOURCE_CREDIBILITY: 0.25,
                ValidationStepType.CONSENSUS: 0.35,
                ValidationStepType.COMPLETENESS: 0.25,
                ValidationStepType.LLM_INFERENCE: 0.15
            }
        )

        # Pipeline 4: Use Case Scenarios (Scenarios Query)
        configs[PipelineType.USE_CASE_SCENARIOS] = create_pipeline_config(
            pipeline_type=PipelineType.USE_CASE_SCENARIOS,
            steps=[
                ValidationStepType.RETRIEVAL,
                ValidationStepType.SOURCE_CREDIBILITY,
                ValidationStepType.COMPLETENESS,
                ValidationStepType.LLM_INFERENCE
            ],
            confidence_weights={
                ValidationStepType.SOURCE_CREDIBILITY: 0.3,
                ValidationStepType.COMPLETENESS: 0.4,
                ValidationStepType.LLM_INFERENCE: 0.3
            }
        )

        # Pipeline 5: Expert Debate (Debate Query)
        configs[PipelineType.EXPERT_DEBATE] = create_pipeline_config(
            pipeline_type=PipelineType.EXPERT_DEBATE,
            steps=[
                ValidationStepType.RETRIEVAL,
                ValidationStepType.SOURCE_CREDIBILITY,
                ValidationStepType.CONSENSUS,
                ValidationStepType.LLM_INFERENCE
            ],
            confidence_weights={
                ValidationStepType.SOURCE_CREDIBILITY: 0.4,
                ValidationStepType.CONSENSUS: 0.4,
                ValidationStepType.LLM_INFERENCE: 0.2
            }
        )

        # Pipeline 6: User Experience (Quotes Query)
        configs[PipelineType.USER_EXPERIENCE] = create_pipeline_config(
            pipeline_type=PipelineType.USER_EXPERIENCE,
            steps=[
                ValidationStepType.RETRIEVAL,
                ValidationStepType.SOURCE_CREDIBILITY,
                ValidationStepType.CONSENSUS,
                ValidationStepType.LLM_INFERENCE
            ],
            confidence_weights={
                ValidationStepType.SOURCE_CREDIBILITY: 0.5,
                ValidationStepType.CONSENSUS: 0.3,
                ValidationStepType.LLM_INFERENCE: 0.2
            }
        )

        return configs

    def _register_validation_steps(self) -> Dict[ValidationStepType, Type]:
        """Register validation step implementations"""

        # Import validation step classes
        from .steps.source_credibility_validator import SourceCredibilityValidator
        from .steps.technical_consistency_validator import TechnicalConsistencyValidator
        from .steps.completeness import CompletenessValidator
        from .steps.consensus import ConsensusValidator
        from .steps.retrieval import RetrievalValidator
        from .steps.llm_inference import LLMInferenceValidator

        return {
            ValidationStepType.RETRIEVAL: RetrievalValidator,
            ValidationStepType.SOURCE_CREDIBILITY: SourceCredibilityValidator,
            ValidationStepType.TECHNICAL_CONSISTENCY: TechnicalConsistencyValidator,
            ValidationStepType.COMPLETENESS: CompletenessValidator,
            ValidationStepType.CONSENSUS: ConsensusValidator,
            ValidationStepType.LLM_INFERENCE: LLMInferenceValidator
        }

    def determine_pipeline_type(self, query_mode: str, query_text: str) -> PipelineType:
        """
        Determine the appropriate validation pipeline based on query characteristics
        """

        # Map query modes to pipeline types
        mode_mapping = {
            "facts": PipelineType.SPECIFICATION_VERIFICATION,
            "features": PipelineType.FEATURE_COMPARISON,
            "tradeoffs": PipelineType.TRADEOFF_ANALYSIS,
            "scenarios": PipelineType.USE_CASE_SCENARIOS,
            "debate": PipelineType.EXPERT_DEBATE,
            "quotes": PipelineType.USER_EXPERIENCE
        }

        pipeline_type = mode_mapping.get(query_mode, PipelineType.SPECIFICATION_VERIFICATION)

        logger.info(f"Selected pipeline type: {pipeline_type.value} for query mode: {query_mode}")
        return pipeline_type

    def get_pipeline_config(self, pipeline_type: PipelineType) -> ValidationPipelineConfig:
        """Get configuration for a specific pipeline type"""
        return self.pipeline_configs.get(pipeline_type)

    async def execute_validation_pipeline(
        self,
        context: ValidationContext,
        pipeline_type: Optional[PipelineType] = None
    ) -> ValidationChainResult:
        """
        Execute a complete validation pipeline
        """

        # Determine pipeline type if not provided
        if pipeline_type is None:
            pipeline_type = self.determine_pipeline_type(
                context.query_mode,
                context.query_text
            )

        # Get pipeline configuration
        config = self.get_pipeline_config(pipeline_type)
        if not config:
            raise ValueError(f"No configuration found for pipeline type: {pipeline_type}")

        logger.info(f"Executing validation pipeline: {config.pipeline_name}")

        # Initialize validation chain result
        chain_result = ValidationChainResult(
            chain_id=f"{pipeline_type.value}_{datetime.now().isoformat()}",
            query_id=context.query_id,
            pipeline_type=pipeline_type,
            overall_status=ValidationStatus.PENDING,
            confidence=ConfidenceBreakdown(total_score=0.0, level=ConfidenceLevel.POOR),
            validation_steps=[],
            started_at=datetime.now(),
            step_progression=[]
        )

        # Execute each validation step
        for step_config in config.steps:
            if not step_config.enabled:
                logger.info(f"Skipping disabled step: {step_config.step_name}")
                continue

            logger.info(f"Executing validation step: {step_config.step_name}")

            # Get step implementation
            step_class = self.validation_steps.get(step_config.step_type)
            if not step_class:
                logger.error(f"No implementation found for step type: {step_config.step_type}")
                continue

            # Create step instance
            step_instance = step_class(step_config, self.meta_validator)

            # Execute step with meta-validation
            try:
                step_result = await step_instance.execute(context)
                chain_result.validation_steps.append(step_result)

                # Update step progression for UI
                status_icon = self._get_status_icon(step_result.status)
                chain_result.step_progression.append(f"{status_icon} {step_result.step_name}")

                # Check for contribution opportunities
                if step_result.contribution_prompt:
                    chain_result.contribution_opportunities.append(step_result.contribution_prompt)
                    chain_result.learning_credits_available += step_result.contribution_prompt.confidence_impact

                logger.info(f"Step {step_config.step_name} completed with status: {step_result.status}")

            except Exception as e:
                logger.error(f"Error executing step {step_config.step_name}: {str(e)}")

                # Create error step result
                error_result = ValidationStepResult(
                    step_id=f"{step_config.step_type.value}_error_{datetime.now().isoformat()}",
                    step_type=step_config.step_type,
                    step_name=step_config.step_name,
                    status=ValidationStatus.FAILED,
                    summary=f"Step execution failed: {str(e)}",
                    started_at=datetime.now(),
                    completed_at=datetime.now()
                )
                chain_result.validation_steps.append(error_result)

        # Calculate final confidence and determine overall status
        chain_result.confidence = self.confidence_calculator.calculate_confidence(
            chain_result.validation_steps,
            config.confidence_weights
        )

        chain_result.overall_status = self._determine_overall_status(chain_result.validation_steps)
        chain_result.completed_at = datetime.now()

        if chain_result.started_at and chain_result.completed_at:
            duration = chain_result.completed_at - chain_result.started_at
            chain_result.total_duration_ms = int(duration.total_seconds() * 1000)

        logger.info(
            f"Validation pipeline completed. "
            f"Overall status: {chain_result.overall_status}, "
            f"Confidence: {chain_result.confidence.total_score:.1f}%"
        )

        return chain_result

    def _get_status_icon(self, status: ValidationStatus) -> str:
        """Get emoji icon for validation status"""
        icons = {
            ValidationStatus.PASSED: "✅",
            ValidationStatus.WARNING: "⚠️",
            ValidationStatus.FAILED: "❌",
            ValidationStatus.UNVERIFIABLE: "🚫",
            ValidationStatus.PENDING: "⏳"
        }
        return icons.get(status, "❓")

    def _determine_overall_status(self, step_results: List[ValidationStepResult]) -> ValidationStatus:
        """Determine overall validation status from step results"""

        if not step_results:
            return ValidationStatus.FAILED

        # Count status types
        status_counts = {}
        for result in step_results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

        total_steps = len(step_results)

        # If any critical step failed, overall is failed
        if status_counts.get(ValidationStatus.FAILED, 0) > 0:
            return ValidationStatus.FAILED

        # If more than half are unverifiable, overall is unverifiable
        if status_counts.get(ValidationStatus.UNVERIFIABLE, 0) > total_steps / 2:
            return ValidationStatus.UNVERIFIABLE

        # If any warnings exist, overall is warning
        if status_counts.get(ValidationStatus.WARNING, 0) > 0:
            return ValidationStatus.WARNING

        # If any unverifiable exist, overall is warning
        if status_counts.get(ValidationStatus.UNVERIFIABLE, 0) > 0:
            return ValidationStatus.WARNING

        # All passed
        return ValidationStatus.PASSED

    async def retry_validation_step(
        self,
        chain_result: ValidationChainResult,
        step_type: ValidationStepType,
        context: ValidationContext
    ) -> Optional[ValidationStepResult]:
        """
        Retry a specific validation step (used after user contributions)
        """

        # Get pipeline configuration
        config = self.get_pipeline_config(chain_result.pipeline_type)
        if not config:
            logger.error(f"No configuration found for pipeline type: {chain_result.pipeline_type}")
            return None

        # Find step configuration
        step_config = None
        for step in config.steps:
            if step.step_type == step_type:
                step_config = step
                break

        if not step_config:
            logger.error(f"No configuration found for step type: {step_type}")
            return None

        # Get step implementation
        step_class = self.validation_steps.get(step_type)
        if not step_class:
            logger.error(f"No implementation found for step type: {step_type}")
            return None

        # Create step instance and execute
        step_instance = step_class(step_config, self.meta_validator)

        try:
            logger.info(f"Retrying validation step: {step_config.step_name}")
            step_result = await step_instance.execute(context)

            # Update retry count
            step_result.retry_count = 1

            logger.info(f"Step retry completed with status: {step_result.status}")
            return step_result

        except Exception as e:
            logger.error(f"Error retrying step {step_config.step_name}: {str(e)}")
            return None

    def get_pipeline_types(self) -> List[PipelineType]:
        """Get all available pipeline types"""
        return list(self.pipeline_configs.keys())

    def get_pipeline_description(self, pipeline_type: PipelineType) -> str:
        """Get description for a pipeline type"""
        config = self.get_pipeline_config(pipeline_type)
        return config.description if config else "Unknown pipeline"