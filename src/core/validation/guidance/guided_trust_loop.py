"""
Guided Trust Loop Implementation
Handles user contributions and learning system
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..models.validation_models import (
    UserContribution, ValidationUpdate, LearningCredit,
    ContributionType, ValidationStepType
)

logger = logging.getLogger(__name__)


class GuidanceEngine:
    """
    Generates user guidance for improving validation
    """

    def __init__(self):
        self.guidance_templates = self._initialize_guidance_templates()

    def _initialize_guidance_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize guidance templates for different validation failures"""

        return {
            "missing_epa_data": {
                "title": "EPA Fuel Economy Data Needed",
                "description": "We need official EPA fuel economy ratings to verify fuel efficiency claims.",
                "suggested_sources": [
                    {
                        "name": "EPA Fuel Economy Guide",
                        "url": "https://www.fueleconomy.gov/feg/findacar.shtml",
                        "description": "Official EPA fuel economy ratings"
                    }
                ],
                "confidence_boost": 25.0
            },
            "missing_manufacturer_specs": {
                "title": "Official Manufacturer Specifications Needed",
                "description": "We need official specifications from the manufacturer to verify technical claims.",
                "suggested_sources": [
                    {
                        "name": "Manufacturer Website",
                        "url_template": "https://www.{manufacturer}.com",
                        "description": "Official vehicle specifications"
                    }
                ],
                "confidence_boost": 20.0
            },
            "insufficient_sources": {
                "title": "Additional Sources Needed",
                "description": "More diverse sources would improve validation reliability.",
                "suggested_sources": [
                    {
                        "name": "Professional Reviews",
                        "description": "Add reviews from automotive publications"
                    },
                    {
                        "name": "Official Documentation",
                        "description": "Include manufacturer or regulatory sources"
                    }
                ],
                "confidence_boost": 15.0
            }
        }

    def generate_guidance_for_failure(
            self,
            failure_type: str,
            context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate specific guidance for a validation failure"""

        template = self.guidance_templates.get(failure_type, self.guidance_templates["insufficient_sources"])

        # Customize template with context
        guidance = template.copy()

        # Replace placeholders
        if "url_template" in str(guidance):
            manufacturer = context.get("manufacturer", "manufacturer")
            guidance["suggested_sources"] = [
                {
                    **source,
                    "url": source.get("url_template", "").format(manufacturer=manufacturer)
                }
                for source in guidance["suggested_sources"]
            ]

        return guidance


class ContributionHandler:
    """
    Handles processing of user contributions
    """

    def __init__(self):
        self.contribution_validators = {
            "url_link": self._validate_url_contribution,
            "text_input": self._validate_text_contribution,
            "file_upload": self._validate_file_contribution
        }

        # Simple in-memory store (in production, use database)
        self.contributions_store = {}
        self.learning_credits_store = {}

    async def process_contribution(
            self,
            job_id: str,
            step_type: str,
            contribution_data: Dict[str, Any],
            original_validation: Any
    ) -> Any:
        """Process a user contribution and return results"""

        logger.info(f"Processing contribution for job {job_id}, step {step_type}")

        # Create contribution record
        contribution = UserContribution(
            contribution_id=f"contrib_{job_id}_{step_type}_{datetime.now().isoformat()}",
            user_id=contribution_data.get("user_id", "anonymous"),
            validation_id=job_id,
            failed_step=ValidationStepType(step_type),
            contribution_type=ContributionType(contribution_data.get("type", "url_link")),
            source_url=contribution_data.get("source_url"),
            text_content=contribution_data.get("text_content"),
            additional_context=contribution_data.get("additional_context"),
            vehicle_scope=contribution_data.get("vehicle_scope", "general"),
            submitted_at=datetime.now()
        )

        # Validate contribution
        validation_result = await self._validate_contribution(contribution)

        if validation_result["valid"]:
            # Accept contribution
            contribution.status = "accepted"
            contribution.processed_at = datetime.now()

            # Store contribution
            self.contributions_store[contribution.contribution_id] = contribution

            # Generate learning credit
            learning_credit = self._generate_learning_credit(contribution, validation_result)
            self.learning_credits_store[learning_credit.credit_id] = learning_credit

            # Create update result
            return ValidationUpdate(
                update_id=f"update_{contribution.contribution_id}",
                original_validation_id=job_id,
                contribution_id=contribution.contribution_id,
                status="validation_updated",
                confidence_improvement=validation_result.get("confidence_improvement", 10.0),
                contribution_accepted=True,
                learning_credit=learning_credit,
                knowledge_base_updates=[f"Added {contribution.contribution_type.value} for {step_type}"],
                future_validation_impact=f"Will improve {step_type} validation for similar queries",
                processed_at=datetime.now()
            )

        else:
            # Reject contribution
            contribution.status = "rejected"
            contribution.rejection_reason = validation_result.get("reason", "Invalid contribution")
            contribution.processed_at = datetime.now()

            return ValidationUpdate(
                update_id=f"update_{contribution.contribution_id}",
                original_validation_id=job_id,
                contribution_id=contribution.contribution_id,
                status="contribution_rejected",
                confidence_improvement=0.0,
                contribution_accepted=False,
                processed_at=datetime.now()
            )

    async def _validate_contribution(self, contribution: UserContribution) -> Dict[str, Any]:
        """Validate a user contribution"""

        contribution_type = contribution.contribution_type
        validator = self.contribution_validators.get(contribution_type.value)

        if not validator:
            return {"valid": False, "reason": "Unknown contribution type"}

        return await validator(contribution)

    async def _validate_url_contribution(self, contribution: UserContribution) -> Dict[str, Any]:
        """Validate URL contribution"""

        url = contribution.source_url
        if not url:
            return {"valid": False, "reason": "No URL provided"}

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            return {"valid": False, "reason": "Invalid URL format"}

        # Check if URL is from a known reliable domain
        reliable_domains = [
            "epa.gov", "nhtsa.gov", "iihs.org",
            "toyota.com", "honda.com", "ford.com", "bmw.com", "mercedes-benz.com",
            "motortrend.com", "caranddriver.com", "edmunds.com"
        ]

        domain = url.split("//")[1].split("/")[0].lower()
        domain = domain.replace("www.", "")

        confidence_improvement = 5.0  # Base improvement

        if any(reliable_domain in domain for reliable_domain in reliable_domains):
            confidence_improvement = 15.0  # Higher improvement for reliable sources

        return {
            "valid": True,
            "confidence_improvement": confidence_improvement,
            "source_quality": "high" if confidence_improvement > 10 else "medium"
        }

    async def _validate_text_contribution(self, contribution: UserContribution) -> Dict[str, Any]:
        """Validate text contribution"""

        text = contribution.text_content
        if not text or len(text.strip()) < 10:
            return {"valid": False, "reason": "Text too short or empty"}

        # Basic quality check
        if len(text) > 1000:
            return {"valid": False, "reason": "Text too long"}

        return {
            "valid": True,
            "confidence_improvement": 8.0,
            "source_quality": "medium"
        }

    async def _validate_file_contribution(self, contribution: UserContribution) -> Dict[str, Any]:
        """Validate file contribution"""

        # For now, accept all file contributions
        # In production, would validate file type, size, content, etc.

        return {
            "valid": True,
            "confidence_improvement": 12.0,
            "source_quality": "medium"
        }

    def _generate_learning_credit(
            self,
            contribution: UserContribution,
            validation_result: Dict[str, Any]
    ) -> LearningCredit:
        """Generate learning credit for accepted contribution"""

        base_points = 10.0
        quality_multiplier = {
            "high": 2.0,
            "medium": 1.5,
            "low": 1.0
        }.get(validation_result.get("source_quality", "medium"), 1.0)

        total_points = base_points * quality_multiplier

        return LearningCredit(
            credit_id=f"credit_{contribution.contribution_id}",
            user_id=contribution.user_id,
            contribution_id=contribution.contribution_id,
            credit_points=total_points,
            credit_category=f"{contribution.failed_step.value}_improvement",
            impact_summary=f"Improved {contribution.failed_step.value} validation",
            benefits=[
                f"Enhanced validation for {contribution.vehicle_scope}",
                f"Contributed {contribution.contribution_type.value} resource",
                f"Increased system knowledge base"
            ],
            validation_improvement=validation_result.get("confidence_improvement", 0.0),
            future_benefit_estimate=f"Will help validate similar vehicles in the future",
            earned_at=datetime.now()
        )

    def get_user_learning_credits(self, user_id: str) -> List[LearningCredit]:
        """Get all learning credits for a user"""

        return [
            credit for credit in self.learning_credits_store.values()
            if credit.user_id == user_id
        ]

    def get_user_total_points(self, user_id: str) -> float:
        """Get total learning points for a user"""

        credits = self.get_user_learning_credits(user_id)
        return sum(credit.credit_points for credit in credits)


class LearningTracker:
    """
    Tracks learning and system improvement from user contributions
    """

    def __init__(self):
        # Simple in-memory tracking (use database in production)
        self.knowledge_base_updates = []
        self.validation_improvements = {}
        self.user_statistics = {}

    def track_knowledge_update(
            self,
            contribution_id: str,
            update_type: str,
            impact_area: str,
            improvement_details: Dict[str, Any]
    ):
        """Track updates to the knowledge base"""

        update_record = {
            "contribution_id": contribution_id,
            "update_type": update_type,
            "impact_area": impact_area,
            "improvement_details": improvement_details,
            "timestamp": datetime.now().isoformat()
        }

        self.knowledge_base_updates.append(update_record)

        # Update validation improvements tracking
        if impact_area not in self.validation_improvements:
            self.validation_improvements[impact_area] = {
                "total_contributions": 0,
                "total_improvement": 0.0,
                "last_updated": datetime.now().isoformat()
            }

        self.validation_improvements[impact_area]["total_contributions"] += 1
        self.validation_improvements[impact_area]["total_improvement"] += improvement_details.get("confidence_boost",
                                                                                                  0.0)
        self.validation_improvements[impact_area]["last_updated"] = datetime.now().isoformat()

    def track_user_contribution(self, user_id: str, contribution: UserContribution, learning_credit: LearningCredit):
        """Track user contribution statistics"""

        if user_id not in self.user_statistics:
            self.user_statistics[user_id] = {
                "total_contributions": 0,
                "total_points": 0.0,
                "contribution_types": {},
                "validation_areas": {},
                "first_contribution": datetime.now().isoformat(),
                "last_contribution": datetime.now().isoformat()
            }

        stats = self.user_statistics[user_id]
        stats["total_contributions"] += 1
        stats["total_points"] += learning_credit.credit_points
        stats["last_contribution"] = datetime.now().isoformat()

        # Track contribution types
        contrib_type = contribution.contribution_type.value
        stats["contribution_types"][contrib_type] = stats["contribution_types"].get(contrib_type, 0) + 1

        # Track validation areas
        validation_area = contribution.failed_step.value
        stats["validation_areas"][validation_area] = stats["validation_areas"].get(validation_area, 0) + 1

    def get_system_learning_summary(self) -> Dict[str, Any]:
        """Get summary of system learning and improvement"""

        return {
            "total_knowledge_updates": len(self.knowledge_base_updates),
            "validation_improvements": self.validation_improvements,
            "total_users_contributing": len(self.user_statistics),
            "total_contributions": sum(stats["total_contributions"] for stats in self.user_statistics.values()),
            "total_points_awarded": sum(stats["total_points"] for stats in self.user_statistics.values()),
            "most_improved_areas": sorted(
                self.validation_improvements.items(),
                key=lambda x: x[1]["total_improvement"],
                reverse=True
            )[:5]
        }

    def get_user_impact_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of a user's impact on the system"""

        stats = self.user_statistics.get(user_id, {})

        if not stats:
            return {"user_found": False}

        return {
            "user_found": True,
            "total_contributions": stats["total_contributions"],
            "total_points": stats["total_points"],
            "contribution_breakdown": stats["contribution_types"],
            "validation_areas_improved": stats["validation_areas"],
            "member_since": stats["first_contribution"],
            "last_contribution": stats["last_contribution"],
            "contribution_rank": self._calculate_user_rank(user_id),
            "impact_level": self._determine_impact_level(stats["total_points"])
        }

    def _calculate_user_rank(self, user_id: str) -> int:
        """Calculate user's rank based on total points"""

        user_points = self.user_statistics.get(user_id, {}).get("total_points", 0)

        # Count users with higher points
        higher_users = sum(
            1 for stats in self.user_statistics.values()
            if stats["total_points"] > user_points
        )

        return higher_users + 1

    def _determine_impact_level(self, total_points: float) -> str:
        """Determine user impact level based on points"""

        if total_points >= 500:
            return "Expert Contributor"
        elif total_points >= 200:
            return "Advanced Contributor"
        elif total_points >= 50:
            return "Active Contributor"
        elif total_points >= 10:
            return "New Contributor"
        else:
            return "Getting Started"


# Global instances
guidance_engine = GuidanceEngine()
contribution_handler = ContributionHandler()
learning_tracker = LearningTracker()