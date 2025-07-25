"""
Meta-Validator: Validation of the Validation
Checks preconditions for validation steps before they execute
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from src.models import (
    ValidationStepType, ValidationContext, PreconditionFailure,
    ValidationGuidance, ContributionPrompt, ContributionType
)

logger = logging.getLogger(__name__)


class PreconditionResult:
    """Result of precondition checking"""

    def __init__(self, status: str, failure_reason: str = "", missing_resources: List[PreconditionFailure] = None):
        self.status = status  # "READY", "PARTIAL", "UNVERIFIABLE"
        self.failure_reason = failure_reason
        self.missing_resources = missing_resources or []
        self.suggestions = []


class MetaValidator:
    """
    Validates that validation steps can execute properly before running them
    """

    def __init__(self):
        self.knowledge_base = self._initialize_knowledge_base()
        self.guidance_generator = ValidationGuidanceGenerator()

    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize knowledge base status tracking"""
        return {
            "epa_database": {
                "available": True,
                "last_updated": datetime.now() - timedelta(days=30),
                "coverage": {
                    "2024": 0.85,  # 85% of 2024 models covered
                    "2023": 0.95,
                    "2022": 0.98
                }
            },
            "manufacturer_specs": {
                "available": True,
                "last_updated": datetime.now() - timedelta(days=15),
                "coverage": {
                    "toyota": 0.90,
                    "honda": 0.88,
                    "ford": 0.85,
                    "gm": 0.82,
                    "bmw": 0.75,
                    "mercedes": 0.78
                }
            },
            "safety_ratings": {
                "available": True,
                "last_updated": datetime.now() - timedelta(days=60),
                "coverage": {
                    "nhtsa": 0.80,
                    "iihs": 0.75
                }
            },
            "source_authority_db": {
                "available": True,
                "last_updated": datetime.now() - timedelta(days=7),
                "domain_count": 1250
            }
        }

    async def check_preconditions(
            self,
            step_type: ValidationStepType,
            context: ValidationContext,
            step_config: Dict[str, Any] = None
    ) -> PreconditionResult:
        """
        Check if a validation step can execute properly
        """

        logger.info(f"Checking preconditions for step: {step_type.value}")

        # Route to specific precondition checker
        if step_type == ValidationStepType.RETRIEVAL:
            return await self._check_retrieval_preconditions(context, step_config)
        elif step_type == ValidationStepType.SOURCE_CREDIBILITY:
            return await self._check_source_credibility_preconditions(context, step_config)
        elif step_type == ValidationStepType.TECHNICAL_CONSISTENCY:
            return await self._check_technical_consistency_preconditions(context, step_config)
        elif step_type == ValidationStepType.COMPLETENESS:
            return await self._check_completeness_preconditions(context, step_config)
        elif step_type == ValidationStepType.CONSENSUS:
            return await self._check_consensus_preconditions(context, step_config)
        elif step_type == ValidationStepType.LLM_INFERENCE:
            return await self._check_llm_inference_preconditions(context, step_config)
        else:
            logger.warning(f"Unknown validation step type: {step_type}")
            return PreconditionResult("READY")

    async def _check_retrieval_preconditions(
            self,
            context: ValidationContext,
            step_config: Dict[str, Any]
    ) -> PreconditionResult:
        """Check preconditions for document retrieval"""

        missing_resources = []

        # Check if documents were actually retrieved
        if not context.documents or len(context.documents) == 0:
            missing_resources.append(PreconditionFailure(
                resource_type="document_retrieval",
                resource_name="Relevant documents",
                failure_reason="No documents found for query",
                impact_description="Cannot perform any validation without source documents",
                suggested_action="Expand search terms or check vector database"
            ))

        # Check minimum document threshold
        elif len(context.documents) < 2:
            missing_resources.append(PreconditionFailure(
                resource_type="document_diversity",
                resource_name="Sufficient document count",
                failure_reason=f"Only {len(context.documents)} document(s) found",
                impact_description="Limited validation capability with single source",
                suggested_action="Add more diverse sources to knowledge base"
            ))

        if missing_resources:
            return PreconditionResult(
                status="UNVERIFIABLE" if len(context.documents) == 0 else "PARTIAL",
                failure_reason="Insufficient document retrieval for reliable validation",
                missing_resources=missing_resources
            )

        return PreconditionResult("READY")

    async def _check_source_credibility_preconditions(
            self,
            context: ValidationContext,
            step_config: Dict[str, Any]
    ) -> PreconditionResult:
        """Check preconditions for source credibility validation"""

        missing_resources = []

        # Check source authority database availability
        auth_db = self.knowledge_base.get("source_authority_db", {})
        if not auth_db.get("available", False):
            missing_resources.append(PreconditionFailure(
                resource_type="authority_database",
                resource_name="Source authority database",
                failure_reason="Authority database is offline or unavailable",
                impact_description="Cannot assess source credibility automatically",
                suggested_action="Manual source evaluation required"
            ))

        # Check data freshness
        last_updated = auth_db.get("last_updated")
        if last_updated and (datetime.now() - last_updated).days > 30:
            missing_resources.append(PreconditionFailure(
                resource_type="data_freshness",
                resource_name="Current authority data",
                failure_reason="Authority database is outdated",
                impact_description="Source credibility scores may be inaccurate",
                suggested_action="Update authority database"
            ))

        # Check minimum document threshold for credibility assessment
        if len(context.documents) < 2:
            missing_resources.append(PreconditionFailure(
                resource_type="source_diversity",
                resource_name="Multiple sources for credibility comparison",
                failure_reason="Need at least 2 sources for credibility assessment",
                impact_description="Cannot compare source reliability",
                suggested_action="Expand search to find additional sources"
            ))

        if missing_resources:
            status = "UNVERIFIABLE" if not auth_db.get("available", False) else "PARTIAL"
            return PreconditionResult(
                status=status,
                failure_reason="Limited source credibility assessment capability",
                missing_resources=missing_resources
            )

        return PreconditionResult("READY")

    async def _check_technical_consistency_preconditions(
            self,
            context: ValidationContext,
            step_config: Dict[str, Any]
    ) -> PreconditionResult:
        """Check preconditions for technical consistency validation"""

        missing_resources = []

        # Extract vehicle information from context
        manufacturer = context.manufacturer
        model = context.model
        year = context.year

        # Check EPA database availability for fuel economy validation
        epa_db = self.knowledge_base.get("epa_database", {})
        if not epa_db.get("available", False):
            missing_resources.append(PreconditionFailure(
                resource_type="epa_database",
                resource_name="EPA fuel economy database",
                failure_reason="EPA database is unavailable",
                impact_description="Cannot verify fuel economy claims",
                suggested_action="Enable EPA database integration"
            ))
        elif year:
            # Check coverage for specific year
            coverage = epa_db.get("coverage", {}).get(str(year), 0)
            if coverage < 0.5:  # Less than 50% coverage
                missing_resources.append(PreconditionFailure(
                    resource_type="epa_coverage",
                    resource_name=f"EPA data for {year} vehicles",
                    failure_reason=f"Limited EPA coverage for {year} ({coverage * 100:.0f}%)",
                    impact_description="May not have fuel economy data for this vehicle",
                    suggested_action=f"Verify fuel economy manually at fueleconomy.gov"
                ))

        # Check manufacturer specification database
        mfg_db = self.knowledge_base.get("manufacturer_specs", {})
        if manufacturer and not mfg_db.get("available", False):
            missing_resources.append(PreconditionFailure(
                resource_type="manufacturer_specs",
                resource_name="Manufacturer specification database",
                failure_reason="Manufacturer specification database unavailable",
                impact_description="Cannot verify technical specifications",
                suggested_action="Check manufacturer website directly"
            ))
        elif manufacturer:
            # Check coverage for specific manufacturer
            mfg_coverage = mfg_db.get("coverage", {}).get(manufacturer.lower(), 0)
            if mfg_coverage < 0.6:  # Less than 60% coverage
                missing_resources.append(PreconditionFailure(
                    resource_type="manufacturer_coverage",
                    resource_name=f"{manufacturer} specification data",
                    failure_reason=f"Limited {manufacturer} coverage ({mfg_coverage * 100:.0f}%)",
                    impact_description="May not have complete technical specifications",
                    suggested_action=f"Visit {manufacturer.lower()}.com for official specs"
                ))

        # Check safety ratings database
        safety_db = self.knowledge_base.get("safety_ratings", {})
        if year and year >= 2020:  # Only check for recent vehicles
            if not safety_db.get("available", False):
                missing_resources.append(PreconditionFailure(
                    resource_type="safety_ratings",
                    resource_name="Safety ratings database",
                    failure_reason="Safety ratings database unavailable",
                    impact_description="Cannot verify safety claims",
                    suggested_action="Check NHTSA and IIHS websites"
                ))

        if missing_resources:
            # Determine severity based on what's missing
            critical_missing = any(
                res.resource_type in ["epa_database", "manufacturer_specs"]
                for res in missing_resources
            )
            status = "UNVERIFIABLE" if critical_missing else "PARTIAL"

            return PreconditionResult(
                status=status,
                failure_reason="Missing critical reference data for technical validation",
                missing_resources=missing_resources
            )

        return PreconditionResult("READY")

    async def _check_completeness_preconditions(
            self,
            context: ValidationContext,
            step_config: Dict[str, Any]
    ) -> PreconditionResult:
        """Check preconditions for completeness validation"""

        missing_resources = []

        # This step primarily validates context completeness,
        # so it mainly needs the context itself

        # Check if we have minimum context for automotive queries
        if not any([context.manufacturer, context.model, context.year]):
            # This isn't necessarily a precondition failure,
            # but rather what the completeness step will validate
            pass

        # Check if documents contain sufficient metadata for completeness assessment
        if not context.documents:
            missing_resources.append(PreconditionFailure(
                resource_type="document_metadata",
                resource_name="Document metadata for completeness assessment",
                failure_reason="No documents available for completeness validation",
                impact_description="Cannot assess context completeness",
                suggested_action="Ensure document retrieval is successful"
            ))

        if missing_resources:
            return PreconditionResult(
                status="UNVERIFIABLE",
                failure_reason="Cannot assess completeness without documents",
                missing_resources=missing_resources
            )

        return PreconditionResult("READY")

    async def _check_consensus_preconditions(
            self,
            context: ValidationContext,
            step_config: Dict[str, Any]
    ) -> PreconditionResult:
        """Check preconditions for consensus validation"""

        missing_resources = []

        # Need multiple sources for consensus
        if len(context.documents) < 3:
            missing_resources.append(PreconditionFailure(
                resource_type="source_diversity",
                resource_name="Multiple sources for consensus analysis",
                failure_reason=f"Only {len(context.documents)} source(s) available",
                impact_description="Cannot assess consensus with limited sources",
                suggested_action="Expand search to find more diverse sources"
            ))

        # Check source diversity (different source types)
        source_types = set()
        for doc in context.documents:
            metadata = doc.get("metadata", {})
            source_type = metadata.get("source_type") or metadata.get("source")
            if source_type:
                source_types.add(source_type)

        if len(source_types) < 2:
            missing_resources.append(PreconditionFailure(
                resource_type="source_type_diversity",
                resource_name="Diverse source types",
                failure_reason="All sources are from similar types",
                impact_description="Limited consensus reliability with homogeneous sources",
                suggested_action="Include sources from different platforms/types"
            ))

        if missing_resources:
            return PreconditionResult(
                status="PARTIAL",
                failure_reason="Limited consensus assessment capability",
                missing_resources=missing_resources
            )

        return PreconditionResult("READY")

    async def _check_llm_inference_preconditions(
            self,
            context: ValidationContext,
            step_config: Dict[str, Any]
    ) -> PreconditionResult:
        """Check preconditions for LLM inference validation"""

        missing_resources = []

        # Check if we have documents to work with
        if not context.documents:
            missing_resources.append(PreconditionFailure(
                resource_type="source_documents",
                resource_name="Documents for LLM inference",
                failure_reason="No documents available for LLM processing",
                impact_description="Cannot generate or validate LLM response",
                suggested_action="Ensure document retrieval is successful"
            ))

        # Check if query is clear enough for inference
        if not context.query_text or len(context.query_text.strip()) < 10:
            missing_resources.append(PreconditionFailure(
                resource_type="query_clarity",
                resource_name="Clear query for inference",
                failure_reason="Query is too short or unclear",
                impact_description="Cannot generate meaningful response",
                suggested_action="Provide more specific query"
            ))

        if missing_resources:
            return PreconditionResult(
                status="UNVERIFIABLE",
                failure_reason="Cannot perform LLM inference",
                missing_resources=missing_resources
            )

        return PreconditionResult("READY")


class ValidationGuidanceGenerator:
    """
    Generates actionable user_guidance for resolving validation precondition failures
    """

    def generate_guidance(
            self,
            failed_step: ValidationStepType,
            missing_resources: List[PreconditionFailure],
            context: ValidationContext
    ) -> ValidationGuidance:
        """Generate specific user_guidance for resolving validation failures"""

        if failed_step == ValidationStepType.TECHNICAL_CONSISTENCY:
            return self._generate_technical_consistency_guidance(missing_resources, context)
        elif failed_step == ValidationStepType.SOURCE_CREDIBILITY:
            return self._generate_source_credibility_guidance(missing_resources, context)
        elif failed_step == ValidationStepType.CONSENSUS:
            return self._generate_consensus_guidance(missing_resources, context)
        else:
            return self._generate_generic_guidance(missing_resources, context)

    def _generate_technical_consistency_guidance(
            self,
            missing_resources: List[PreconditionFailure],
            context: ValidationContext
    ) -> ValidationGuidance:
        """Generate user_guidance for technical consistency validation failures"""

        vehicle_info = self._format_vehicle_info(context)
        suggestions = []

        for resource in missing_resources:
            if "epa" in resource.resource_type.lower():
                suggestions.append({
                    "type": "official_source",
                    "title": "EPA Fuel Economy Database",
                    "description": f"Find official fuel economy ratings for {vehicle_info}",
                    "url": "https://www.fueleconomy.gov/feg/findacar.shtml",
                    "search_hint": f"Search for {vehicle_info} on EPA.gov",
                    "contribution_type": ContributionType.URL_LINK,
                    "confidence_boost": 25.0
                })

            if "manufacturer" in resource.resource_type.lower():
                manufacturer = context.manufacturer
                if manufacturer:
                    suggestions.append({
                        "type": "manufacturer_official",
                        "title": f"Official {manufacturer.title()} Specifications",
                        "description": f"Get official specs from {manufacturer} website",
                        "url": f"https://www.{manufacturer.lower()}.com",
                        "search_hint": f"Look for {vehicle_info} specifications",
                        "contribution_type": ContributionType.URL_LINK,
                        "confidence_boost": 20.0
                    })

        return ValidationGuidance(
            guidance_type="missing_technical_data",
            primary_message="We need official technical specifications to verify claims.",
            suggestions=suggestions,
            user_prompt="Help us verify this information by providing official sources:",
            learning_opportunity="Your contribution will help validate this and similar vehicles in the future.",
            estimated_confidence_boost=sum(s.get("confidence_boost", 0) for s in suggestions)
        )

    def _generate_source_credibility_guidance(
            self,
            missing_resources: List[PreconditionFailure],
            context: ValidationContext
    ) -> ValidationGuidance:
        """Generate user_guidance for source credibility validation failures"""

        suggestions = [
            {
                "type": "authority_update",
                "title": "Manual Source Evaluation",
                "description": "Help evaluate source authority manually",
                "action": "Rate the reliability of sources found",
                "contribution_type": ContributionType.TEXT_INPUT,
                "confidence_boost": 15.0
            }
        ]

        return ValidationGuidance(
            guidance_type="source_credibility_assessment",
            primary_message="Unable to automatically assess source credibility.",
            suggestions=suggestions,
            user_prompt="Please help evaluate source reliability:",
            learning_opportunity="Your expertise helps improve our source evaluation.",
            estimated_confidence_boost=15.0
        )

    def _generate_consensus_guidance(
            self,
            missing_resources: List[PreconditionFailure],
            context: ValidationContext
    ) -> ValidationGuidance:
        """Generate user_guidance for consensus validation failures"""

        vehicle_info = self._format_vehicle_info(context)
        suggestions = [
            {
                "type": "additional_sources",
                "title": "Find Additional Sources",
                "description": f"Add more diverse sources about {vehicle_info}",
                "contribution_type": ContributionType.URL_LINK,
                "confidence_boost": 20.0
            }
        ]

        return ValidationGuidance(
            guidance_type="insufficient_consensus_sources",
            primary_message="Need more diverse sources for reliable consensus analysis.",
            suggestions=suggestions,
            user_prompt="Help us find additional sources:",
            learning_opportunity="More sources improve validation for everyone.",
            estimated_confidence_boost=20.0
        )

    def _generate_generic_guidance(
            self,
            missing_resources: List[PreconditionFailure],
            context: ValidationContext
    ) -> ValidationGuidance:
        """Generate generic user_guidance for validation failures"""

        return ValidationGuidance(
            guidance_type="generic_failure",
            primary_message="Validation step cannot complete due to missing requirements.",
            suggestions=[],
            user_prompt="Please check the requirements and try again.",
            learning_opportunity="Your input helps improve our validation system."
        )

    def _format_vehicle_info(self, context: ValidationContext) -> str:
        """Format vehicle information for display"""
        parts = []
        if context.year:
            parts.append(str(context.year))
        if context.manufacturer:
            parts.append(context.manufacturer)
        if context.model:
            parts.append(context.model)

        return " ".join(parts) if parts else "this vehicle"

    def create_contribution_prompt(
            self,
            guidance: ValidationGuidance,
            step_type: ValidationStepType
    ) -> ContributionPrompt:
        """Create a user contribution prompt from user_guidance"""

        return ContributionPrompt(
            needed_resource_type=guidance.guidance_type,
            specific_need_description=guidance.primary_message,
            contribution_types=[ContributionType.URL_LINK, ContributionType.TEXT_INPUT],
            examples=[s.get("search_hint", "") for s in guidance.suggestions if s.get("search_hint")],
            confidence_impact=guidance.estimated_confidence_boost,
            future_benefit_description=guidance.learning_opportunity
        )