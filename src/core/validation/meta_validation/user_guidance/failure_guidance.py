"""
Guidance Engine for Validation Failures
Generates user-friendly user_guidance for resolving validation issues
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class GuidanceEngine:
    """
    Generates user user_guidance for improving validation
    """

    def __init__(self):
        self.guidance_templates = self._initialize_guidance_templates()

    def _initialize_guidance_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize user_guidance templates for different validation failures"""

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
            },
            "low_source_authority": {
                "title": "Higher Authority Sources Needed",
                "description": "Current sources have limited authority. Official sources would improve validation.",
                "suggested_sources": [
                    {
                        "name": "Official Government Data",
                        "description": "EPA, NHTSA, or other regulatory sources"
                    },
                    {
                        "name": "Manufacturer Official Data",
                        "description": "Direct from vehicle manufacturer"
                    }
                ],
                "confidence_boost": 20.0
            }
        }

    def generate_guidance_for_failure(self, failure_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific user_guidance for a validation failure"""

        # Determine failure type from context
        if "epa" in failure_type.lower() or "fuel" in failure_type.lower():
            template_key = "missing_epa_data"
        elif "manufacturer" in failure_type.lower() or "official" in failure_type.lower():
            template_key = "missing_manufacturer_specs"
        elif "authority" in failure_type.lower() or "credibility" in failure_type.lower():
            template_key = "low_source_authority"
        else:
            template_key = "insufficient_sources"

        template = self.guidance_templates.get(template_key, self.guidance_templates["insufficient_sources"])

        # Customize template with context
        guidance = template.copy()

        # Replace placeholders
        if "url_template" in str(guidance):
            manufacturer = context.get("manufacturer", "manufacturer")
            guidance["suggested_sources"] = [
                {
                    **source,
                    "url": source.get("url_template", "").format(manufacturer=manufacturer.lower())
                }
                for source in guidance["suggested_sources"]
            ]

        # Add context-specific information
        guidance["context"] = {
            "failed_step": failure_type,
            "vehicle_info": self._extract_vehicle_info(context),
            "failure_details": context.get("failure_reason", "Validation could not complete")
        }

        return guidance

    def _extract_vehicle_info(self, context: Dict[str, Any]) -> str:
        """Extract vehicle information for display."""

        manufacturer = context.get("manufacturer", "")
        model = context.get("model", "")
        year = context.get("year", "")

        parts = [str(part) for part in [year, manufacturer, model] if part]
        return " ".join(parts) if parts else "this vehicle"