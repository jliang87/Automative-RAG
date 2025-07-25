"""
Technical Consistency Validator
Validates technical specifications consistency for automotive information
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.models import (
    ValidationStepResult, ValidationStatus, ValidationStepType,
    ValidationContext, ValidationWarning
)
from .steps_readiness_checker import MetaValidator, PreconditionResult

logger = logging.getLogger(__name__)


class TechnicalConsistencyValidator:
    """
    Validates consistency of technical specifications across sources
    """

    def __init__(self, step_config: Dict[str, Any], meta_validator: MetaValidator):
        self.step_config = step_config
        self.meta_validator = meta_validator
        self.step_type = ValidationStepType.TECHNICAL_CONSISTENCY

        # Reference data for validation
        self.reference_data = self._initialize_reference_data()
        self.physics_constraints = self._initialize_physics_constraints()

    def _initialize_reference_data(self) -> Dict[str, Any]:
        """Initialize reference technical data for validation"""
        return {
            "epa_fuel_economy": {
                # Sample EPA data for validation
                "2023_toyota_camry": {"city_mpg": 22, "highway_mpg": 32, "combined_mpg": 26},
                "2023_honda_accord": {"city_mpg": 23, "highway_mpg": 34, "combined_mpg": 27},
                "2023_ford_f150": {"city_mpg": 18, "highway_mpg": 24, "combined_mpg": 20},
            },
            "manufacturer_specs": {
                "toyota_camry_2023": {
                    "engine": "2.5L 4-cylinder",
                    "horsepower": 203,
                    "torque": 184,
                    "transmission": "8-speed automatic",
                    "weight": 3340
                },
                "honda_accord_2023": {
                    "engine": "1.5L turbo 4-cylinder",
                    "horsepower": 192,
                    "torque": 192,
                    "transmission": "CVT",
                    "weight": 3131
                }
            },
            "safety_ratings": {
                "2023_toyota_camry": {"nhtsa_overall": 5, "iihs_top_safety_pick": True},
                "2023_honda_accord": {"nhtsa_overall": 5, "iihs_top_safety_pick": True}
            }
        }

    def _initialize_physics_constraints(self) -> Dict[str, Dict[str, Any]]:
        """Initialize physics constraints for validation"""
        return {
            "fuel_economy": {
                "reasonable_ranges": {
                    "city_mpg": {"min": 8, "max": 60},
                    "highway_mpg": {"min": 10, "max": 80},
                    "combined_mpg": {"min": 9, "max": 70}
                },
                "physics_rules": [
                    "highway_mpg >= city_mpg",  # Highway typically better than city
                    "city_mpg * 0.8 <= combined_mpg <= highway_mpg * 0.8 + city_mpg * 0.2"
                    # Combined is weighted average
                ]
            },
            "engine_specs": {
                "reasonable_ranges": {
                    "horsepower": {"min": 100, "max": 800},
                    "torque": {"min": 100, "max": 700},
                    "displacement": {"min": 1.0, "max": 8.0}  # Liters
                },
                "physics_rules": [
                    "horsepower > 0",
                    "torque > 0"
                ]
            },
            "vehicle_dimensions": {
                "reasonable_ranges": {
                    "weight": {"min": 2000, "max": 8000},  # lbs
                    "length": {"min": 140, "max": 250},  # inches
                    "width": {"min": 60, "max": 90},  # inches
                    "height": {"min": 50, "max": 85}  # inches
                }
            }
        }

    async def execute(self, context: ValidationContext) -> ValidationStepResult:
        """Execute technical consistency validation"""

        start_time = datetime.now()

        # Check preconditions
        precondition_result = await self.meta_validator.check_preconditions(
            self.step_type, context, self.step_config
        )

        if precondition_result.status != "READY":
            return self._create_unverifiable_result(start_time, precondition_result)

        try:
            # Perform technical consistency validation
            result = await self._perform_validation(context)
            result.completed_at = datetime.now()
            result.duration_ms = int((result.completed_at - start_time).total_seconds() * 1000)

            return result

        except Exception as e:
            logger.error(f"Technical consistency validation failed: {str(e)}")
            return self._create_error_result(start_time, str(e))

    async def _perform_validation(self, context: ValidationContext) -> ValidationStepResult:
        """Perform the actual technical consistency validation"""

        documents = context.documents
        warnings = []
        sources_analyzed = []

        # Extract technical specifications from all documents
        extracted_specs = []
        for i, doc in enumerate(documents):
            specs = self._extract_technical_specs(doc, i)
            if specs:
                extracted_specs.append(specs)
                sources_analyzed.append(specs.get("source_id", f"document_{i}"))

        # Perform consistency checks
        consistency_results = {
            "fuel_economy_consistency": self._check_fuel_economy_consistency(extracted_specs),
            "engine_specs_consistency": self._check_engine_specs_consistency(extracted_specs),
            "safety_ratings_consistency": self._check_safety_ratings_consistency(extracted_specs),
            "physics_validation": self._validate_physics_constraints(extracted_specs),
            "cross_reference_validation": self._validate_against_reference_data(extracted_specs, context)
        }

        # Analyze results and generate warnings
        consistency_issues = []
        physics_issues = []
        reference_mismatches = []

        for check_type, results in consistency_results.items():
            if results.get("status") == "inconsistent":
                issues = results.get("issues", [])

                if check_type == "physics_validation":
                    physics_issues.extend(issues)
                elif check_type == "cross_reference_validation":
                    reference_mismatches.extend(issues)
                else:
                    consistency_issues.extend(issues)

                # Generate warnings
                for issue in issues:
                    severity = "critical" if issue.get("severity") == "high" else "caution"
                    warnings.append(ValidationWarning(
                        category=check_type,
                        severity=severity,
                        message=issue.get("message", "Technical inconsistency detected"),
                        explanation=issue.get("explanation", ""),
                        suggestion=issue.get("suggestion", "Verify with official sources")
                    ))

        # Calculate overall consistency score
        total_checks = len(consistency_results)
        passed_checks = sum(1 for r in consistency_results.values() if r.get("status") == "consistent")
        consistency_score = passed_checks / total_checks if total_checks > 0 else 0.0

        # Determine validation status and confidence impact
        if consistency_score >= 0.8 and not physics_issues:
            status = ValidationStatus.PASSED
            confidence_impact = 10.0 + (consistency_score - 0.8) * 25  # 10-15 point boost
        elif consistency_score >= 0.6:
            status = ValidationStatus.WARNING
            confidence_impact = 0.0 + (consistency_score - 0.6) * 25  # 0-5 point boost
        else:
            status = ValidationStatus.WARNING
            confidence_impact = -5.0 + (consistency_score * 10)  # Up to -5 point penalty

        # Apply additional penalties for physics violations
        if physics_issues:
            confidence_impact -= len(physics_issues) * 2.0  # -2 points per physics issue

        # Apply penalties for reference mismatches
        if reference_mismatches:
            confidence_impact -= len(reference_mismatches) * 1.0  # -1 point per mismatch

        # Build summary
        summary = (f"Analyzed {len(extracted_specs)} sources for technical consistency. "
                   f"Consistency score: {consistency_score:.2f}/1.0. "
                   f"Issues found: {len(consistency_issues + physics_issues + reference_mismatches)}")

        # Build detailed results
        details = {
            "specifications_extracted": len(extracted_specs),
            "consistency_score": consistency_score,
            "checks_performed": total_checks,
            "checks_passed": passed_checks,
            "consistency_results": consistency_results,
            "physics_validated": len(physics_issues) == 0,
            "reference_data_matches": len(reference_mismatches) == 0,
            "official_sources_used": sum(1 for spec in extracted_specs
                                         if spec.get("source_type") == "official"),
            "specifications_found": self._summarize_specifications(extracted_specs)
        }

        return ValidationStepResult(
            step_id=f"technical_consistency_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Technical Consistency Validation",
            status=status,
            confidence_impact=confidence_impact,
            summary=summary,
            details=details,
            started_at=start_time,
            warnings=warnings,
            sources_used=sources_analyzed
        )

    def _extract_technical_specs(self, doc: Dict[str, Any], doc_index: int) -> Optional[Dict[str, Any]]:
        """Extract technical specifications from a document"""

        content = doc.get("content", "")
        metadata = doc.get("metadata", {})

        if not content:
            return None

        specs = {
            "source_id": metadata.get("url", f"document_{doc_index}"),
            "source_type": self._determine_source_type(metadata),
            "extracted_specs": {}
        }

        # Extract fuel economy specifications
        fuel_economy = self._extract_fuel_economy(content)
        if fuel_economy:
            specs["extracted_specs"]["fuel_economy"] = fuel_economy

        # Extract engine specifications
        engine_specs = self._extract_engine_specs(content)
        if engine_specs:
            specs["extracted_specs"]["engine"] = engine_specs

        # Extract safety ratings
        safety_ratings = self._extract_safety_ratings(content)
        if safety_ratings:
            specs["extracted_specs"]["safety"] = safety_ratings

        # Extract vehicle dimensions
        dimensions = self._extract_dimensions(content)
        if dimensions:
            specs["extracted_specs"]["dimensions"] = dimensions

        return specs if specs["extracted_specs"] else None

    def _extract_fuel_economy(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract fuel economy data from content"""

        fuel_economy = {}

        # Look for MPG patterns
        city_match = re.search(r'city.*?(\d+)\s*mpg', content, re.IGNORECASE)
        if city_match:
            fuel_economy["city_mpg"] = int(city_match.group(1))

        highway_match = re.search(r'highway.*?(\d+)\s*mpg', content, re.IGNORECASE)
        if highway_match:
            fuel_economy["highway_mpg"] = int(highway_match.group(1))

        combined_match = re.search(r'combined.*?(\d+)\s*mpg', content, re.IGNORECASE)
        if combined_match:
            fuel_economy["combined_mpg"] = int(combined_match.group(1))

        return fuel_economy if fuel_economy else None

    def _extract_engine_specs(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract engine specifications from content"""

        engine_specs = {}

        # Look for horsepower
        hp_match = re.search(r'(\d+)\s*(?:hp|horsepower)', content, re.IGNORECASE)
        if hp_match:
            engine_specs["horsepower"] = int(hp_match.group(1))

        # Look for torque
        torque_match = re.search(r'(\d+)\s*(?:lb-ft|lbft|ft-lbs|torque)', content, re.IGNORECASE)
        if torque_match:
            engine_specs["torque"] = int(torque_match.group(1))

        # Look for engine displacement
        displacement_match = re.search(r'(\d+\.?\d*)\s*(?:l|liter)', content, re.IGNORECASE)
        if displacement_match:
            engine_specs["displacement"] = float(displacement_match.group(1))

        return engine_specs if engine_specs else None

    def _extract_safety_ratings(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract safety ratings from content"""

        safety_ratings = {}

        # Look for NHTSA ratings
        nhtsa_match = re.search(r'nhtsa.*?(\d+)\s*star', content, re.IGNORECASE)
        if nhtsa_match:
            safety_ratings["nhtsa_overall"] = int(nhtsa_match.group(1))

        # Look for IIHS awards
        if re.search(r'iihs.*?top.*?safety.*?pick', content, re.IGNORECASE):
            safety_ratings["iihs_top_safety_pick"] = True

        return safety_ratings if safety_ratings else None

    def _extract_dimensions(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract vehicle dimensions from content"""

        dimensions = {}

        # Look for weight
        weight_match = re.search(r'(?:curb\s+)?weight.*?(\d+,?\d+)\s*(?:lbs?|pounds)', content, re.IGNORECASE)
        if weight_match:
            weight_str = weight_match.group(1).replace(',', '')
            dimensions["weight"] = int(weight_str)

        return dimensions if dimensions else None

    def _determine_source_type(self, metadata: Dict[str, Any]) -> str:
        """Determine the type of source for authority weighting"""

        url = metadata.get("url", "").lower()

        if any(domain in url for domain in ['.gov', 'epa.', 'nhtsa.']):
            return "regulatory"
        elif any(domain in url for domain in ['toyota.com', 'honda.com', 'ford.com', 'bmw.com']):
            return "official"
        elif any(domain in url for domain in ['edmunds.com', 'kbb.com', 'motortrend.com']):
            return "professional"
        else:
            return "other"

    def _check_fuel_economy_consistency(self, extracted_specs: List[Dict]) -> Dict[str, Any]:
        """Check consistency of fuel economy data across sources"""

        fuel_economy_data = []
        for spec in extracted_specs:
            if "fuel_economy" in spec["extracted_specs"]:
                fuel_economy_data.append(spec["extracted_specs"]["fuel_economy"])

        if len(fuel_economy_data) < 2:
            return {"status": "insufficient_data", "message": "Not enough fuel economy data to compare"}

        issues = []

        # Check for large discrepancies
        for mpg_type in ["city_mpg", "highway_mpg", "combined_mpg"]:
            values = [data[mpg_type] for data in fuel_economy_data if mpg_type in data]
            if len(values) >= 2:
                min_val, max_val = min(values), max(values)
                if max_val - min_val > 5:  # More than 5 MPG difference
                    issues.append({
                        "message": f"Large discrepancy in {mpg_type}",
                        "explanation": f"Values range from {min_val} to {max_val} MPG",
                        "suggestion": "Verify with EPA database",
                        "severity": "medium"
                    })

        return {
            "status": "inconsistent" if issues else "consistent",
            "issues": issues,
            "data_points": len(fuel_economy_data)
        }

    def _check_engine_specs_consistency(self, extracted_specs: List[Dict]) -> Dict[str, Any]:
        """Check consistency of engine specifications across sources"""

        engine_data = []
        for spec in extracted_specs:
            if "engine" in spec["extracted_specs"]:
                engine_data.append(spec["extracted_specs"]["engine"])

        if len(engine_data) < 2:
            return {"status": "insufficient_data", "message": "Not enough engine data to compare"}

        issues = []

        # Check horsepower consistency
        hp_values = [data["horsepower"] for data in engine_data if "horsepower" in data]
        if len(hp_values) >= 2:
            min_hp, max_hp = min(hp_values), max(hp_values)
            if max_hp - min_hp > 20:  # More than 20 HP difference
                issues.append({
                    "message": "Horsepower values inconsistent",
                    "explanation": f"Values range from {min_hp} to {max_hp} HP",
                    "suggestion": "Check manufacturer specifications",
                    "severity": "medium"
                })

        # Check torque consistency
        torque_values = [data["torque"] for data in engine_data if "torque" in data]
        if len(torque_values) >= 2:
            min_torque, max_torque = min(torque_values), max(torque_values)
            if max_torque - min_torque > 15:  # More than 15 lb-ft difference
                issues.append({
                    "message": "Torque values inconsistent",
                    "explanation": f"Values range from {min_torque} to {max_torque} lb-ft",
                    "suggestion": "Check manufacturer specifications",
                    "severity": "medium"
                })

        return {
            "status": "inconsistent" if issues else "consistent",
            "issues": issues,
            "data_points": len(engine_data)
        }

    def _check_safety_ratings_consistency(self, extracted_specs: List[Dict]) -> Dict[str, Any]:
        """Check consistency of safety ratings across sources"""

        safety_data = []
        for spec in extracted_specs:
            if "safety" in spec["extracted_specs"]:
                safety_data.append(spec["extracted_specs"]["safety"])

        if len(safety_data) < 2:
            return {"status": "insufficient_data", "message": "Not enough safety data to compare"}

        issues = []

        # Check NHTSA rating consistency
        nhtsa_values = [data["nhtsa_overall"] for data in safety_data if "nhtsa_overall" in data]
        if len(nhtsa_values) >= 2 and len(set(nhtsa_values)) > 1:
            issues.append({
                "message": "NHTSA ratings inconsistent",
                "explanation": f"Different ratings found: {set(nhtsa_values)}",
                "suggestion": "Verify with NHTSA database",
                "severity": "high"
            })

        return {
            "status": "inconsistent" if issues else "consistent",
            "issues": issues,
            "data_points": len(safety_data)
        }

    def _validate_physics_constraints(self, extracted_specs: List[Dict]) -> Dict[str, Any]:
        """Validate extracted specifications against physics constraints"""

        issues = []

        for spec in extracted_specs:
            source_id = spec["source_id"]

            # Validate fuel economy physics
            if "fuel_economy" in spec["extracted_specs"]:
                fuel_data = spec["extracted_specs"]["fuel_economy"]
                fuel_issues = self._validate_fuel_economy_physics(fuel_data, source_id)
                issues.extend(fuel_issues)

            # Validate engine specs physics
            if "engine" in spec["extracted_specs"]:
                engine_data = spec["extracted_specs"]["engine"]
                engine_issues = self._validate_engine_physics(engine_data, source_id)
                issues.extend(engine_issues)

        return {
            "status": "inconsistent" if issues else "consistent",
            "issues": issues,
            "physics_checks_performed": len(extracted_specs)
        }

    def _validate_fuel_economy_physics(self, fuel_data: Dict, source_id: str) -> List[Dict]:
        """Validate fuel economy data against physics constraints"""

        issues = []
        constraints = self.physics_constraints["fuel_economy"]

        # Check reasonable ranges
        for mpg_type, value in fuel_data.items():
            if mpg_type in constraints["reasonable_ranges"]:
                range_constraint = constraints["reasonable_ranges"][mpg_type]
                if not (range_constraint["min"] <= value <= range_constraint["max"]):
                    issues.append({
                        "message": f"Unrealistic {mpg_type}: {value}",
                        "explanation": f"Value outside reasonable range ({range_constraint['min']}-{range_constraint['max']})",
                        "suggestion": "Verify this specification",
                        "severity": "high",
                        "source": source_id
                    })

        # Check physics rules
        if "city_mpg" in fuel_data and "highway_mpg" in fuel_data:
            if fuel_data["highway_mpg"] < fuel_data["city_mpg"]:
                issues.append({
                    "message": "Highway MPG lower than city MPG",
                    "explanation": f"Highway: {fuel_data['highway_mpg']}, City: {fuel_data['city_mpg']}",
                    "suggestion": "This is unusual - verify the values",
                    "severity": "medium",
                    "source": source_id
                })

        return issues

    def _validate_engine_physics(self, engine_data: Dict, source_id: str) -> List[Dict]:
        """Validate engine specifications against physics constraints"""

        issues = []
        constraints = self.physics_constraints["engine_specs"]

        # Check reasonable ranges
        for spec_type, value in engine_data.items():
            if spec_type in constraints["reasonable_ranges"]:
                range_constraint = constraints["reasonable_ranges"][spec_type]
                if not (range_constraint["min"] <= value <= range_constraint["max"]):
                    issues.append({
                        "message": f"Unrealistic {spec_type}: {value}",
                        "explanation": f"Value outside reasonable range ({range_constraint['min']}-{range_constraint['max']})",
                        "suggestion": "Verify this specification",
                        "severity": "high",
                        "source": source_id
                    })

        return issues

    def _validate_against_reference_data(self, extracted_specs: List[Dict], context: ValidationContext) -> Dict[
        str, Any]:
        """Validate extracted specifications against reference databases"""

        issues = []
        matches_found = 0

        # Try to match vehicle from context
        vehicle_key = self._build_vehicle_key(context)

        for spec in extracted_specs:
            source_id = spec["source_id"]

            # Check fuel economy against EPA data
            if "fuel_economy" in spec["extracted_specs"] and vehicle_key:
                epa_data = self.reference_data["epa_fuel_economy"].get(vehicle_key)
                if epa_data:
                    matches_found += 1
                    fuel_issues = self._compare_with_epa_data(
                        spec["extracted_specs"]["fuel_economy"], epa_data, source_id
                    )
                    issues.extend(fuel_issues)

            # Check engine specs against manufacturer data
            if "engine" in spec["extracted_specs"] and vehicle_key:
                mfg_data = self.reference_data["manufacturer_specs"].get(vehicle_key)
                if mfg_data:
                    matches_found += 1
                    engine_issues = self._compare_with_manufacturer_data(
                        spec["extracted_specs"]["engine"], mfg_data, source_id
                    )
                    issues.extend(engine_issues)

        return {
            "status": "inconsistent" if issues else "consistent",
            "issues": issues,
            "reference_matches": matches_found
        }

    def _build_vehicle_key(self, context: ValidationContext) -> Optional[str]:
        """Build a vehicle key for reference data lookup"""

        if context.year and context.manufacturer and context.model:
            return f"{context.year}_{context.manufacturer.lower()}_{context.model.lower()}"
        return None

    def _compare_with_epa_data(self, extracted_fuel: Dict, epa_data: Dict, source_id: str) -> List[Dict]:
        """Compare extracted fuel economy with EPA reference data"""

        issues = []
        tolerance = 2  # Allow 2 MPG tolerance

        for mpg_type in ["city_mpg", "highway_mpg", "combined_mpg"]:
            if mpg_type in extracted_fuel and mpg_type in epa_data:
                extracted_value = extracted_fuel[mpg_type]
                epa_value = epa_data[mpg_type]

                if abs(extracted_value - epa_value) > tolerance:
                    issues.append({
                        "message": f"EPA data mismatch for {mpg_type}",
                        "explanation": f"Source: {extracted_value} MPG, EPA: {epa_value} MPG",
                        "suggestion": "Verify against official EPA database",
                        "severity": "medium",
                        "source": source_id
                    })

        return issues

    def _compare_with_manufacturer_data(self, extracted_engine: Dict, mfg_data: Dict, source_id: str) -> List[Dict]:
        """Compare extracted engine specs with manufacturer reference data"""

        issues = []

        # Compare horsepower (allow 5% tolerance)
        if "horsepower" in extracted_engine and "horsepower" in mfg_data:
            extracted_hp = extracted_engine["horsepower"]
            mfg_hp = mfg_data["horsepower"]
            tolerance = mfg_hp * 0.05

            if abs(extracted_hp - mfg_hp) > tolerance:
                issues.append({
                    "message": "Horsepower mismatch with manufacturer data",
                    "explanation": f"Source: {extracted_hp} HP, Manufacturer: {mfg_hp} HP",
                    "suggestion": "Verify against official manufacturer specifications",
                    "severity": "medium",
                    "source": source_id
                })

        return issues

    def _summarize_specifications(self, extracted_specs: List[Dict]) -> Dict[str, Any]:
        """Summarize all extracted specifications"""

        summary = {
            "fuel_economy_sources": 0,
            "engine_spec_sources": 0,
            "safety_rating_sources": 0,
            "dimension_sources": 0
        }

        for spec in extracted_specs:
            if "fuel_economy" in spec["extracted_specs"]:
                summary["fuel_economy_sources"] += 1
            if "engine" in spec["extracted_specs"]:
                summary["engine_spec_sources"] += 1
            if "safety" in spec["extracted_specs"]:
                summary["safety_rating_sources"] += 1
            if "dimensions" in spec["extracted_specs"]:
                summary["dimension_sources"] += 1

        return summary

    def _create_unverifiable_result(self, start_time: datetime,
                                    precondition_result: PreconditionResult) -> ValidationStepResult:
        """Create result for unverifiable validation"""

        return ValidationStepResult(
            step_id=f"technical_consistency_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Technical Consistency Validation",
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
                    message="Cannot validate technical consistency",
                    explanation=precondition_result.failure_reason,
                    suggestion="Check EPA database and manufacturer data availability"
                )
            ]
        )

    def _create_error_result(self, start_time: datetime, error_message: str) -> ValidationStepResult:
        """Create result for validation errors"""

        return ValidationStepResult(
            step_id=f"technical_consistency_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Technical Consistency Validation",
            status=ValidationStatus.FAILED,
            confidence_impact=-10.0,
            summary=f"Validation failed: {error_message}",
            details={"error": error_message},
            started_at=start_time,
            completed_at=datetime.now(),
            warnings=[
                ValidationWarning(
                    category="validation_error",
                    severity="critical",
                    message="Technical consistency validation encountered an error",
                    explanation=error_message,
                    suggestion="Check logs and retry validation"
                )
            ]
        )