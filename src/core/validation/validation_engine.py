"""
Main Validation Engine
Orchestrates the complete validation framework and integrates with job processing
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from .models.validation_models import (
    ValidationContext, ValidationChainResult, PipelineType,
    ValidationStepResult, ValidationStatus
)
from .pipeline_manager import PipelineManager
from .guidance.guidance_engine import GuidanceEngine
from .guidance.contribution_handler import ContributionHandler

logger = logging.getLogger(__name__)


class ValidationEngine:
    """
    Main validation engine that coordinates all validation activities
    """

    def __init__(self):
        self.pipeline_manager = PipelineManager()
        self.guidance_engine = GuidanceEngine()
        self.contribution_handler = ContributionHandler()

        # Cache for validation results
        self.validation_cache = {}

    async def validate_documents(
            self,
            documents: List[Dict[str, Any]],
            query: str,
            query_mode: str,
            metadata_filter: Optional[Dict] = None,
            job_id: Optional[str] = None
    ) -> ValidationChainResult:
        """
        Main entry point for document validation
        This replaces the automotive_fact_check_documents function
        """

        logger.info(f"Starting document validation for query: {query[:100]}...")

        # Create validation context
        context = self._create_validation_context(
            documents=documents,
            query=query,
            query_mode=query_mode,
            metadata_filter=metadata_filter,
            job_id=job_id
        )

        # Determine pipeline type
        pipeline_type = self.pipeline_manager.determine_pipeline_type(query_mode, query)

        # Execute validation pipeline
        validation_result = await self.pipeline_manager.execute_validation_pipeline(
            context=context,
            pipeline_type=pipeline_type
        )

        # Cache result for potential retries
        if job_id:
            self.validation_cache[job_id] = validation_result

        logger.info(
            f"Document validation completed. "
            f"Confidence: {validation_result.confidence.total_score:.1f}%, "
            f"Status: {validation_result.overall_status}"
        )

        return validation_result

    async def validate_answer(
            self,
            answer: str,
            documents: List[Dict[str, Any]],
            query: str,
            query_mode: str,
            job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate LLM-generated answer
        This replaces the automotive_fact_check_answer function
        """

        logger.info(f"Starting answer validation for query: {query[:100]}...")

        # Create validation context for answer validation
        context = self._create_validation_context(
            documents=documents,
            query=query,
            query_mode=query_mode,
            job_id=job_id
        )

        # Add answer to context
        context.processing_metadata["generated_answer"] = answer

        # Use LLM inference pipeline for answer validation
        pipeline_type = PipelineType.SPECIFICATION_VERIFICATION  # Default for answer validation

        validation_result = await self.pipeline_manager.execute_validation_pipeline(
            context=context,
            pipeline_type=pipeline_type
        )

        # Format result for compatibility with existing code
        answer_validation = self._format_answer_validation_result(validation_result, answer)

        logger.info(f"Answer validation completed. Confidence: {validation_result.confidence.total_score:.1f}%")

        return answer_validation

    def format_automotive_warnings_for_user(self, validation_result: ValidationChainResult) -> str:
        """
        Format validation warnings for user display
        This replaces the format_automotive_warnings_for_user function
        """

        warnings = []

        # Collect warnings from all validation steps
        for step_result in validation_result.validation_steps:
            for warning in step_result.warnings:
                if warning.severity in ["critical", "caution"]:
                    warnings.append(f"⚠️ {warning.message}: {warning.explanation}")

        # Add overall confidence warning if low
        if validation_result.confidence.total_score < 70:
            warnings.append(
                f"ℹ️ Confidence Score: {validation_result.confidence.total_score:.0f}% - "
                "Consider verifying information with additional sources"
            )

        # Format as footnotes
        if warnings:
            warning_text = "\n\n" + "---" + "\n"
            warning_text += "**Validation Notes:**\n"
            warning_text += "\n".join(warnings)
            return warning_text

        return ""

    def get_automotive_validation_summary(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get validation summary for documents
        This replaces the get_automotive_validation_summary function
        """

        # Simple summary for documents without full validation
        summary = {
            "total_documents": len(documents),
            "documents_with_warnings": 0,
            "validation_applied": True,
            "domain_specific": "automotive"
        }

        # Count documents with potential issues
        for doc in documents:
            metadata = doc.get("metadata", {})
            if metadata.get("automotive_warnings") or metadata.get("validation_status") == "warning":
                summary["documents_with_warnings"] += 1

        return summary

    async def process_user_contribution(
            self,
            job_id: str,
            step_type: str,
            contribution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process user contribution and retry validation
        This enables the guided trust loop functionality
        """

        logger.info(f"Processing user contribution for job {job_id}, step {step_type}")

        # Get original validation result
        original_result = self.validation_cache.get(job_id)
        if not original_result:
            return {
                "success": False,
                "error": "Original validation result not found"
            }

        # Process contribution
        contribution_result = await self.contribution_handler.process_contribution(
            job_id=job_id,
            step_type=step_type,
            contribution_data=contribution_data,
            original_validation=original_result
        )

        if contribution_result.contribution_accepted:
            # Retry the specific validation step
            updated_context = self._update_context_with_contribution(
                original_result, contribution_result
            )

            retry_result = await self.pipeline_manager.retry_validation_step(
                chain_result=original_result,
                step_type=step_type,
                context=updated_context
            )

            if retry_result:
                # Update cached result
                self._update_validation_result(original_result, retry_result)
                self.validation_cache[job_id] = original_result

                return {
                    "success": True,
                    "validation_updated": True,
                    "new_confidence": original_result.confidence.total_score,
                    "learning_credit": contribution_result.learning_credit
                }

        return {
            "success": False,
            "error": "Contribution could not be processed"
        }

    def _create_validation_context(
            self,
            documents: List[Dict[str, Any]],
            query: str,
            query_mode: str,
            metadata_filter: Optional[Dict] = None,
            job_id: Optional[str] = None
    ) -> ValidationContext:
        """Create validation context from input parameters"""

        # Extract vehicle context from query and documents
        vehicle_context = self._extract_vehicle_context_from_query(query)
        if not vehicle_context and documents:
            vehicle_context = self._extract_vehicle_context_from_documents(documents)

        context = ValidationContext(
            query_id=job_id or f"query_{datetime.now().isoformat()}",
            query_text=query,
            query_mode=query_mode,
            documents=documents,
            retrieval_metadata=metadata_filter or {}
        )

        # Add vehicle context if found
        if vehicle_context:
            context.manufacturer = vehicle_context.get("manufacturer")
            context.model = vehicle_context.get("model")
            context.year = vehicle_context.get("year")
            context.trim = vehicle_context.get("trim")

        return context

    def _extract_vehicle_context_from_query(self, query: str) -> Dict[str, Any]:
        """Extract vehicle information from query text"""

        import re

        context = {}
        query_lower = query.lower()

        # Extract year
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            context["year"] = int(year_match.group(1))

        # Extract common manufacturers
        manufacturers = [
            "toyota", "honda", "ford", "chevrolet", "nissan", "hyundai", "kia",
            "bmw", "mercedes", "audi", "volkswagen", "subaru", "mazda", "lexus",
            "acura", "infiniti", "cadillac", "lincoln", "buick", "gmc",
            "吉利", "比亚迪", "长城", "蔚来", "理想", "小鹏"
        ]

        for manufacturer in manufacturers:
            if manufacturer in query_lower:
                context["manufacturer"] = manufacturer
                break

        # Extract common models (this could be expanded significantly)
        models = [
            "camry", "accord", "civic", "corolla", "altima", "elantra",
            "sonata", "malibu", "fusion", "impala", "3 series", "c-class",
            "a4", "jetta", "outback", "cx-5", "星越", "汉", "唐", "model 3"
        ]

        for model in models:
            if model in query_lower:
                context["model"] = model
                break

        return context

    def _extract_vehicle_context_from_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract vehicle information from document metadata"""

        context = {}

        for doc in documents:
            metadata = doc.get("metadata", {})

            # Try different metadata field names
            if not context.get("manufacturer"):
                manufacturer = metadata.get("manufacturer") or metadata.get("brand")
                if manufacturer:
                    context["manufacturer"] = str(manufacturer).lower()

            if not context.get("model"):
                model = metadata.get("vehicleModel") or metadata.get("model")
                if model:
                    context["model"] = str(model)

            if not context.get("year"):
                year = metadata.get("modelYear") or metadata.get("year")
                if year:
                    try:
                        context["year"] = int(year)
                    except (ValueError, TypeError):
                        pass

            # Stop if we have enough context
            if len(context) >= 3:
                break

        return context

    def _format_answer_validation_result(
            self,
            validation_result: ValidationChainResult,
            answer: str
    ) -> Dict[str, Any]:
        """Format validation result for answer validation compatibility"""

        return {
            "automotive_confidence": validation_result.confidence.level.value,
            "confidence_score": validation_result.confidence.total_score,
            "answer_warnings": [
                warning.message for step in validation_result.validation_steps
                for warning in step.warnings if warning.severity in ["critical", "caution"]
            ],
            "source_warnings": [
                f"Source issue in {step.step_name}" for step in validation_result.validation_steps
                if step.status in [ValidationStatus.WARNING, ValidationStatus.FAILED]
            ],
            "validation_details": {
                "pipeline_type": validation_result.pipeline_type.value,
                "validation_steps": len(validation_result.validation_steps),
                "verification_coverage": validation_result.confidence.verification_coverage,
                "step_results": [
                    {
                        "step": step.step_name,
                        "status": step.status.value,
                        "confidence_impact": step.confidence_impact
                    }
                    for step in validation_result.validation_steps
                ]
            }
        }

    def _update_context_with_contribution(
            self,
            original_result: ValidationChainResult,
            contribution_result: Any
    ) -> ValidationContext:
        """Update validation context with user contribution data"""

        # This would be implemented based on the specific contribution handler
        # For now, return a placeholder context
        return ValidationContext(
            query_id=original_result.query_id,
            query_text="",  # Would be extracted from original result
            query_mode="facts",
            documents=[]
        )

    def _update_validation_result(
            self,
            original_result: ValidationChainResult,
            new_step_result: ValidationStepResult
    ):
        """Update validation result with new step result"""

        # Find and replace the step result
        for i, step in enumerate(original_result.validation_steps):
            if step.step_type == new_step_result.step_type:
                original_result.validation_steps[i] = new_step_result
                break

        # Recalculate confidence
        pipeline_config = self.pipeline_manager.get_pipeline_config(original_result.pipeline_type)
        if pipeline_config:
            original_result.confidence = self.pipeline_manager.confidence_calculator.calculate_confidence(
                original_result.validation_steps,
                pipeline_config.confidence_weights
            )

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics about validation engine performance"""

        cached_results = len(self.validation_cache)

        # Calculate average confidence from cached results
        if cached_results > 0:
            confidences = [result.confidence.total_score for result in self.validation_cache.values()]
            avg_confidence = sum(confidences) / len(confidences)
        else:
            avg_confidence = 0.0

        return {
            "cached_validations": cached_results,
            "average_confidence": avg_confidence,
            "available_pipelines": len(self.pipeline_manager.get_pipeline_types()),
            "engine_status": "operational"
        }


# Global validation engine instance
validation_engine = ValidationEngine()


# ============================================================================
# Compatibility Functions for Existing Code
# ============================================================================

async def automotive_fact_check_documents(
        documents: List[Tuple[Any, float]]
) -> List[Tuple[Any, float]]:
    """
    Compatibility function for existing code
    Validates documents and returns them with updated metadata
    """

    # Convert input format
    doc_list = []
    for doc, score in documents:
        doc_dict = {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "relevance_score": score
        }
        doc_list.append(doc_dict)

    # Perform validation (simplified for compatibility)
    validation_result = await validation_engine.validate_documents(
        documents=doc_list,
        query="automotive information validation",
        query_mode="facts"
    )

    # Add validation metadata to documents
    for i, (doc, score) in enumerate(documents):
        doc.metadata["automotive_validation"] = {
            "validated": True,
            "confidence": validation_result.confidence.total_score,
            "validation_id": validation_result.chain_id
        }

        # Add warnings if any
        warnings = []
        for step in validation_result.validation_steps:
            warnings.extend([w.message for w in step.warnings])

        if warnings:
            doc.metadata["automotive_warnings"] = warnings

    return documents


async def automotive_fact_check_answer(
        answer: str,
        documents: List[Any]
) -> Dict[str, Any]:
    """
    Compatibility function for existing code
    Validates LLM answer against documents
    """

    # Convert documents to expected format
    doc_list = []
    for doc in documents:
        if hasattr(doc, 'page_content'):
            doc_dict = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
        else:
            doc_dict = doc
        doc_list.append(doc_dict)

    # Validate answer
    return await validation_engine.validate_answer(
        answer=answer,
        documents=doc_list,
        query="answer validation",
        query_mode="facts"
    )


def format_automotive_warnings_for_user(validation_results: Dict[str, Any]) -> str:
    """
    Compatibility function for existing code
    """

    warnings = validation_results.get("answer_warnings", [])
    source_warnings = validation_results.get("source_warnings", [])

    all_warnings = warnings + source_warnings

    if all_warnings:
        warning_text = "\n\n" + "---" + "\n"
        warning_text += "**Validation Notes:**\n"
        for warning in all_warnings:
            warning_text += f"⚠️ {warning}\n"
        return warning_text

    return ""


def get_automotive_validation_summary(documents: List[Any]) -> Dict[str, Any]:
    """
    Compatibility function for existing code
    """

    return validation_engine.get_automotive_validation_summary(documents)