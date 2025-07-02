"""
Technical Consistency Validation Step
Validates automotive claims against known technical constraints and official data
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple

from ..models.validation_models import (
    ValidationStepResult, ValidationStatus, ValidationContext,
    create_validation_step_result
)
from ..meta_validator import PreconditionResult
from .base_validation_step import BaseValidationStep

logger = logging.getLogger(__name__)


class TechnicalConsistencyValidator(BaseValidationStep):
    """
    Validates technical claims against automotive constraints and reference data
    """

    def __init__(self, config, meta_validator):
        super().__init__(config, meta_validator)
        self.reference_data = self._initialize_reference_data()
        self.constraint_checker = ConstraintChecker()

    def _initialize_reference_data(self) -> Dict[str, Any]:
        """Initialize automotive reference data"""

        return {
            "epa_data": {
                # Sample EPA fuel economy data
                "2024": {
                    "toyota_camry": {
                        "le": {"city_mpg": 28, "highway_mpg": 39, "combined_mpg": 32},
                        "xle": {"city_mpg": 28, "highway_mpg": 39, "combined_mpg": 32},
                        "hybrid_le": {"city_mpg": 51, "highway_mpg": 53, "combined_mpg": 52}
                    },
                    "honda_accord": {
                        "lx": {"city_mpg": 32, "highway_mpg": 42, "combined_mpg": 36},
                        "sport": {"city_mpg": 30, "highway_mpg": 38, "combined_mpg": 33},
                        "hybrid": {"city_mpg": 48, "highway_mpg": 48, "combined_mpg": 48}
                    },
                    "bmw_3series": {
                        "330i": {"city_mpg": 26, "highway_mpg": 36, "combined_mpg": 30},
                        "m340i": {"city_mpg": 22, "highway_mpg": 30, "combined_mpg": 25}
                    }
                },
                "2023": {
                    "toyota_camry": {
                        "le": {"city_mpg": 28, "highway_mpg": 39, "combined_mpg": 32},
                        "hybrid_le": {"city_mpg": 51, "highway_mpg": 53, "combined_mpg": 52}
                    }
                }
            },
            "manufacturer_specs": {
                "toyota": {
                    "2024": {
                        "camry": {
                            "engines": ["2.5L I4", "2.5L Hybrid I4"],
                            "transmissions": ["8-speed automatic", "CVT"],
                            "msrp_range": (25000, 35000),
                            "body_style": "sedan",
                            "safety_rating": 5
                        }
                    }
                },
                "honda": {
                    "2024": {
                        "accord": {
                            "engines": ["1.5L Turbo I4", "2.0L Turbo I4", "2.0L Hybrid I4"],
                            "transmissions": ["CVT", "10-speed automatic"],
                            "msrp_range": (27000, 38000),
                            "body_style": "sedan",
                            "safety_rating": 5
                        }
                    }
                },
                "bmw": {
                    "2024": {
                        "3series": {
                            "engines": ["2.0L Turbo I4", "3.0L Turbo I6"],
                            "transmissions": ["8-speed automatic"],
                            "msrp_range": (35000, 55000),
                            "body_style": "sedan",
                            "safety_rating": 5
                        }
                    }
                }
            },
            "safety_ratings": {
                "nhtsa": {
                    "2024_toyota_camry": {"overall": 5, "frontal": 5, "side": 5, "rollover": 5},
                    "2024_honda_accord": {"overall": 5, "frontal": 5, "side": 5, "rollover": 5},
                    "2024_bmw_3series": {"overall": 5, "frontal": 4, "side": 5, "rollover": 5}
                },
                "iihs": {
                    "2024_toyota_camry": {"top_safety_pick": True, "plus_award": False},
                    "2024_honda_accord": {"top_safety_pick": True, "plus_award": True}
                }
            }
        }

    async def _execute_validation(
            self,
            context: ValidationContext,
            precondition_result: PreconditionResult
    ) -> ValidationStepResult:
        """Execute technical consistency validation"""

        documents = context.documents
        vehicle_context = self._extract_vehicle_context(context)

        if not documents:
            return create_validation_step_result(
                step_type=self.step_type,
                status=ValidationStatus.FAILED,
                summary="No documents available for technical validation",
                confidence_impact=0.0
            )

        # Extract technical claims from documents
        technical_claims = self._extract_technical_claims(documents)

        # Validate claims against reference data
        validation_results = self._validate_technical_claims(technical_claims, vehicle_context)

        # Check physics and engineering constraints
        constraint_results = self.constraint_checker.check_constraints(technical_claims, vehicle_context)

        # Combine results
        combined_results = self._combine_validation_results(validation_results, constraint_results)

        # Determine overall status and confidence
        status, confidence_impact = self._determine_technical_status(combined_results)

        # Create result
        result = create_validation_step_result(
            step_type=self.step_type,
            status=status,
            summary=self._create_technical_summary(combined_results),
            confidence_impact=confidence_impact
        )

        # Add detailed results
        result.details = {
            "claims_analyzed": len(technical_claims),
            "reference_data_used": validation_results.get("reference_sources", []),
            "physics_checks": constraint_results,
            "validation_results": validation_results,
            "combined_results": combined_results,
            "vehicle_context": vehicle_context
        }

        # Add warnings
        self._add_technical_warnings(result, combined_results)

        return result

    def _extract_technical_claims(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract technical claims from document content"""

        claims = []

        # Patterns for extracting technical information
        patterns = {
            "fuel_economy": [
                r"(\d+\.?\d*)\s*mpg",
                r"油耗.*?(\d+\.?\d*)\s*[升l]",
                r"(\d+\.?\d*)\s*[升l]/100km",
                r"fuel economy.*?(\d+\.?\d*)"
            ],
            "horsepower": [
                r"(\d+)\s*hp",
                r"(\d+)\s*bhp",
                r"(\d+)\s*马力",
                r"(\d+)\s*匹"
            ],
            "price": [
                r"\$(\d+,?\d+)",
                r"(\d+)\s*万",
                r"price.*?\$?(\d+,?\d+)",
                r"msrp.*?\$?(\d+,?\d+)"
            ],
            "engine": [
                r"(\d+\.?\d*)[lL]\s*(V\d+|I\d+|turbo|hybrid)?",
                r"(\d+\.?\d*)\s*升.*?(发动机|引擎)",
                r"engine.*?(\d+\.?\d*)[lL]"
            ],
            "acceleration": [
                r"0-60.*?(\d+\.?\d*)\s*sec",
                r"0-100.*?(\d+\.?\d*)\s*sec",
                r"百公里加速.*?(\d+\.?\d*)\s*秒"
            ]
        }

        for doc in documents:
            content = doc.get('content', '') + ' ' + str(doc.get('metadata', {}))

            for claim_type, claim_patterns in patterns.items():
                for pattern in claim_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        value = match.group(1)
                        try:
                            # Clean and convert value
                            numeric_value = self._clean_numeric_value(value, claim_type)
                            if numeric_value is not None:
                                claims.append({
                                    "type": claim_type,
                                    "value": numeric_value,
                                    "raw_text": match.group(0),
                                    "source_doc": doc.get('metadata', {}).get('url', 'unknown'),
                                    "context": content[max(0, match.start() - 50):match.end() + 50]
                                })
                        except ValueError:
                            continue

        # Deduplicate similar claims
        return self._deduplicate_claims(claims)

    def _clean_numeric_value(self, value: str, claim_type: str) -> Optional[float]:
        """Clean and convert string value to numeric"""

        if not value:
            return None

        # Remove commas and spaces
        cleaned = value.replace(',', '').replace(' ', '')

        try:
            numeric = float(cleaned)

            # Sanity checks based on claim type
            if claim_type == "fuel_economy" and (numeric < 5 or numeric > 150):
                return None
            elif claim_type == "horsepower" and (numeric < 50 or numeric > 1000):
                return None
            elif claim_type == "price" and (numeric < 10000 or numeric > 500000):
                return None
            elif claim_type == "engine" and (numeric < 0.5 or numeric > 10.0):
                return None
            elif claim_type == "acceleration" and (numeric < 2.0 or numeric > 20.0):
                return None

            return numeric

        except ValueError:
            return None

    def _deduplicate_claims(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or very similar claims"""

        deduplicated = []

        for claim in claims:
            # Check if we already have a very similar claim
            duplicate = False
            for existing in deduplicated:
                if (claim["type"] == existing["type"] and
                        abs(claim["value"] - existing["value"]) < 0.1):
                    duplicate = True
                    break

            if not duplicate:
                deduplicated.append(claim)

        return deduplicated

    def _validate_technical_claims(
            self,
            claims: List[Dict[str, Any]],
            vehicle_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate claims against reference data"""

        validation_results = {
            "validated_claims": [],
            "reference_sources": [],
            "validation_coverage": 0.0,
            "accuracy_score": 0.0
        }

        manufacturer = vehicle_context.get('manufacturer', '').lower()
        model = vehicle_context.get('model', '').lower()
        year = vehicle_context.get('year')

        validated_count = 0
        total_accuracy = 0.0

        for claim in claims:
            claim_validation = self._validate_single_claim(claim, manufacturer, model, year)
            validation_results["validated_claims"].append(claim_validation)

            if claim_validation["reference_found"]:
                validated_count += 1
                total_accuracy += claim_validation["accuracy_score"]

                reference_source = claim_validation.get("reference_source")
                if reference_source and reference_source not in validation_results["reference_sources"]:
                    validation_results["reference_sources"].append(reference_source)

        # Calculate coverage and accuracy
        if claims:
            validation_results["validation_coverage"] = validated_count / len(claims)

        if validated_count > 0:
            validation_results["accuracy_score"] = total_accuracy / validated_count

        return validation_results

    def _validate_single_claim(
            self,
            claim: Dict[str, Any],
            manufacturer: str,
            model: str,
            year: Optional[int]
    ) -> Dict[str, Any]:
        """Validate a single technical claim"""

        claim_type = claim["type"]
        claim_value = claim["value"]

        validation = {
            "claim": claim,
            "reference_found": False,
            "reference_value": None,
            "accuracy_score": 0.0,
            "deviation_percent": 0.0,
            "reference_source": None,
            "validation_status": "unverified"
        }

        # Fuel economy validation
        if claim_type == "fuel_economy" and year and manufacturer and model:
            epa_data = self._get_epa_data(manufacturer, model, year)
            if epa_data:
                # Compare against EPA combined MPG
                reference_mpg = epa_data.get("combined_mpg")
                if reference_mpg:
                    validation["reference_found"] = True
                    validation["reference_value"] = reference_mpg
                    validation["reference_source"] = "EPA fuel economy database"

                    # Calculate accuracy
                    deviation = abs(claim_value - reference_mpg) / reference_mpg
                    validation["deviation_percent"] = deviation * 100

                    if deviation <= 0.05:  # Within 5%
                        validation["accuracy_score"] = 1.0
                        validation["validation_status"] = "accurate"
                    elif deviation <= 0.15:  # Within 15%
                        validation["accuracy_score"] = 0.7
                        validation["validation_status"] = "reasonable"
                    elif deviation <= 0.30:  # Within 30%
                        validation["accuracy_score"] = 0.3
                        validation["validation_status"] = "questionable"
                    else:
                        validation["accuracy_score"] = 0.0
                        validation["validation_status"] = "inaccurate"

        # Price validation
        elif claim_type == "price" and year and manufacturer and model:
            spec_data = self._get_manufacturer_specs(manufacturer, model, year)
            if spec_data and "msrp_range" in spec_data:
                min_price, max_price = spec_data["msrp_range"]
                validation["reference_found"] = True
                validation["reference_value"] = f"${min_price:,} - ${max_price:,}"
                validation["reference_source"] = f"{manufacturer.title()} official specifications"

                if min_price <= claim_value <= max_price:
                    validation["accuracy_score"] = 1.0
                    validation["validation_status"] = "accurate"
                elif claim_value < min_price * 0.8 or claim_value > max_price * 1.2:
                    validation["accuracy_score"] = 0.0
                    validation["validation_status"] = "inaccurate"
                else:
                    validation["accuracy_score"] = 0.5
                    validation["validation_status"] = "reasonable"

        # Safety rating validation
        elif claim_type == "safety_rating" and year and manufacturer and model:
            nhtsa_data = self._get_safety_rating(manufacturer, model, year, "nhtsa")
            if nhtsa_data:
                reference_rating = nhtsa_data.get("overall")
                if reference_rating:
                    validation["reference_found"] = True
                    validation["reference_value"] = reference_rating
                    validation["reference_source"] = "NHTSA safety ratings"

                    if claim_value == reference_rating:
                        validation["accuracy_score"] = 1.0
                        validation["validation_status"] = "accurate"
                    else:
                        validation["accuracy_score"] = 0.0
                        validation["validation_status"] = "inaccurate"

        return validation

    def _get_epa_data(self, manufacturer: str, model: str, year: int) -> Optional[Dict[str, Any]]:
        """Get EPA data for specific vehicle"""

        epa_data = self.reference_data.get("epa_data", {})
        year_data = epa_data.get(str(year), {})

        # Create key from manufacturer and model
        vehicle_key = f"{manufacturer}_{model}".replace(" ", "_").lower()

        vehicle_data = year_data.get(vehicle_key)
        if vehicle_data:
            # Return base trim data if available
            if isinstance(vehicle_data, dict):
                # Try common base trims first
                for trim in ["le", "lx", "base", "s"]:
                    if trim in vehicle_data:
                        return vehicle_data[trim]
                # Return first available trim
                first_trim = next(iter(vehicle_data.values()))
                if isinstance(first_trim, dict):
                    return first_trim

        return None

    def _get_manufacturer_specs(self, manufacturer: str, model: str, year: int) -> Optional[Dict[str, Any]]:
        """Get manufacturer specifications for specific vehicle"""

        spec_data = self.reference_data.get("manufacturer_specs", {})
        mfg_data = spec_data.get(manufacturer.lower(), {})
        year_data = mfg_data.get(str(year), {})

        return year_data.get(model.lower())

    def _get_safety_rating(self, manufacturer: str, model: str, year: int, source: str) -> Optional[Dict[str, Any]]:
        """Get safety rating data"""

        safety_data = self.reference_data.get("safety_ratings", {})
        source_data = safety_data.get(source.lower(), {})

        vehicle_key = f"{year}_{manufacturer}_{model}".replace(" ", "_").lower()
        return source_data.get(vehicle_key)

    def _combine_validation_results(
            self,
            validation_results: Dict[str, Any],
            constraint_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine reference validation and constraint checking results"""

        return {
            "reference_validation": validation_results,
            "constraint_validation": constraint_results,
            "overall_accuracy": self._calculate_overall_accuracy(validation_results, constraint_results),
            "physics_validated": constraint_results.get("physics_checks_passed", 0) > 0,
            "official_sources_used": len(validation_results.get("reference_sources", [])),
            "total_validations": len(validation_results.get("validated_claims", [])),
            "validation_summary": self._create_validation_summary(validation_results, constraint_results)
        }

    def _calculate_overall_accuracy(
            self,
            validation_results: Dict[str, Any],
            constraint_results: Dict[str, Any]
    ) -> float:
        """Calculate overall technical accuracy score"""

        # Reference validation accuracy (weighted 70%)
        ref_accuracy = validation_results.get("accuracy_score", 0.0) * 0.7

        # Constraint validation accuracy (weighted 30%)
        constraint_accuracy = 0.0
        physics_checks = constraint_results.get("physics_checks", [])
        if physics_checks:
            passed_checks = sum(1 for check in physics_checks if check.get("passed", False))
            constraint_accuracy = (passed_checks / len(physics_checks)) * 0.3

        return ref_accuracy + constraint_accuracy

    def _create_validation_summary(
            self,
            validation_results: Dict[str, Any],
            constraint_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """Create summary of validation results"""

        summary = {}

        # Reference validation summary
        validated_claims = validation_results.get("validated_claims", [])
        if validated_claims:
            accurate_count = sum(1 for claim in validated_claims
                                 if claim.get("validation_status") == "accurate")
            summary["reference_validation"] = f"{accurate_count}/{len(validated_claims)} claims accurately validated"

        # Constraint validation summary
        physics_checks = constraint_results.get("physics_checks", [])
        if physics_checks:
            passed_count = sum(1 for check in physics_checks if check.get("passed", False))
            summary["constraint_validation"] = f"{passed_count}/{len(physics_checks)} physics checks passed"

        return summary

    def _determine_technical_status(self, combined_results: Dict[str, Any]) -> Tuple[ValidationStatus, float]:
        """Determine validation status and confidence impact"""

        overall_accuracy = combined_results.get("overall_accuracy", 0.0)
        official_sources = combined_results.get("official_sources_used", 0)
        physics_validated = combined_results.get("physics_validated", False)

        # Calculate confidence impact
        confidence_impact = 0.0

        # Base accuracy contribution
        confidence_impact += overall_accuracy * 20.0

        # Official source bonus
        confidence_impact += min(15.0, official_sources * 5.0)

        # Physics validation bonus
        if physics_validated:
            confidence_impact += 10.0

        # Determine status
        if overall_accuracy >= 0.8 and official_sources > 0:
            status = ValidationStatus.PASSED
        elif overall_accuracy >= 0.6:
            status = ValidationStatus.WARNING
        elif overall_accuracy >= 0.3:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAILED

        return status, confidence_impact

    def _create_technical_summary(self, combined_results: Dict[str, Any]) -> str:
        """Create human-readable summary"""

        accuracy = combined_results.get("overall_accuracy", 0.0)
        official_sources = combined_results.get("official_sources_used", 0)
        total_validations = combined_results.get("total_validations", 0)
        physics_validated = combined_results.get("physics_validated", False)

        summary_parts = []

        if total_validations > 0:
            summary_parts.append(f"Validated {total_validations} technical claims")
            summary_parts.append(f"{accuracy:.1%} accuracy")

        if official_sources > 0:
            summary_parts.append(f"{official_sources} official source(s)")

        if physics_validated:
            summary_parts.append("physics constraints verified")

        return ", ".join(summary_parts) if summary_parts else "No technical validation performed"

    def _add_technical_warnings(self, result: ValidationStepResult, combined_results: Dict[str, Any]):
        """Add technical validation warnings"""

        accuracy = combined_results.get("overall_accuracy", 0.0)
        official_sources = combined_results.get("official_sources_used", 0)

        # Low accuracy warning
        if accuracy < 0.6:
            self._add_warning(
                result,
                category="low_technical_accuracy",
                severity="caution",
                message="Technical claims have limited accuracy",
                explanation=f"Overall accuracy: {accuracy:.1%}",
                suggestion="Verify claims with official manufacturer or EPA sources"
            )

        # No official sources warning
        if official_sources == 0:
            self._add_warning(
                result,
                category="no_official_sources",
                severity="info",
                message="No official sources used for technical validation",
                explanation="Validation based on secondary sources only",
                suggestion="Consult manufacturer specifications or EPA data"
            )

        # Physics constraint violations
        constraint_results = combined_results.get("constraint_validation", {})
        failed_checks = [check for check in constraint_results.get("physics_checks", [])
                         if not check.get("passed", True)]

        if failed_checks:
            for check in failed_checks:
                self._add_warning(
                    result,
                    category="physics_constraint_violation",
                    severity="critical",
                    message=f"Physics constraint violation: {check.get('name')}",
                    explanation=check.get("explanation", ""),
                    suggestion="Verify claim against known engineering limits"
                )


class ConstraintChecker:
    """
    Checks automotive claims against physics and engineering constraints
    """

    def __init__(self):
        self.constraints = self._initialize_constraints()

    def _initialize_constraints(self) -> Dict[str, Any]:
        """Initialize physics and engineering constraints"""

        return {
            "fuel_economy": {
                "min_realistic_mpg": 8,  # Heavy trucks/supercars
                "max_realistic_mpg": 120,  # Best hybrids
                "sedan_range": (15, 60),
                "suv_range": (12, 45),
                "hybrid_bonus": 1.5  # Hybrids can be 50% more efficient
            },
            "horsepower": {
                "min_realistic_hp": 70,  # Small economy cars
                "max_realistic_hp": 800,  # High-performance cars
                "power_to_weight": {
                    "economy": (0.08, 0.15),  # HP per pound
                    "performance": (0.15, 0.25),
                    "supercar": (0.25, 0.4)
                }
            },
            "acceleration": {
                "min_realistic_0_60": 2.0,  # Fastest production cars
                "max_realistic_0_60": 15.0,  # Slowest cars
                "power_correlation": True  # More power should mean faster acceleration
            },
            "price": {
                "min_new_car": 15000,
                "max_reasonable": 300000,
                "luxury_threshold": 50000,
                "supercar_threshold": 100000
            }
        }

    def check_constraints(
            self,
            claims: List[Dict[str, Any]],
            vehicle_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check all claims against constraints"""

        results = {
            "physics_checks": [],
            "physics_checks_passed": 0,
            "constraint_violations": [],
            "overall_physics_score": 0.0
        }

        # Group claims by type for cross-validation
        claims_by_type = {}
        for claim in claims:
            claim_type = claim["type"]
            if claim_type not in claims_by_type:
                claims_by_type[claim_type] = []
            claims_by_type[claim_type].append(claim)

        # Check individual constraints
        for claim_type, type_claims in claims_by_type.items():
            for claim in type_claims:
                check_result = self._check_single_constraint(claim, vehicle_context)
                if check_result:
                    results["physics_checks"].append(check_result)
                    if check_result.get("passed", False):
                        results["physics_checks_passed"] += 1
                    else:
                        results["constraint_violations"].append(check_result)

        # Cross-validate related claims
        cross_validation = self._cross_validate_claims(claims_by_type, vehicle_context)
        results["physics_checks"].extend(cross_validation)

        # Calculate overall physics score
        total_checks = len(results["physics_checks"])
        if total_checks > 0:
            results["overall_physics_score"] = results["physics_checks_passed"] / total_checks

        return results

    def _check_single_constraint(
            self,
            claim: Dict[str, Any],
            vehicle_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check a single claim against constraints"""

        claim_type = claim["type"]
        claim_value = claim["value"]

        if claim_type == "fuel_economy":
            return self._check_fuel_economy_constraint(claim_value, vehicle_context)
        elif claim_type == "horsepower":
            return self._check_horsepower_constraint(claim_value, vehicle_context)
        elif claim_type == "acceleration":
            return self._check_acceleration_constraint(claim_value, vehicle_context)
        elif claim_type == "price":
            return self._check_price_constraint(claim_value, vehicle_context)

        return None

    def _check_fuel_economy_constraint(self, mpg: float, vehicle_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check fuel economy against physics constraints"""

        constraints = self.constraints["fuel_economy"]

        # Basic range check
        if mpg < constraints["min_realistic_mpg"] or mpg > constraints["max_realistic_mpg"]:
            return {
                "name": "fuel_economy_range",
                "passed": False,
                "explanation": f"MPG {mpg} outside realistic range ({constraints['min_realistic_mpg']}-{constraints['max_realistic_mpg']})",
                "severity": "critical"
            }

        # Vehicle type specific check
        # This would need more sophisticated vehicle type detection
        # For now, use basic checks

        if mpg > 60 and "hybrid" not in str(vehicle_context).lower():
            return {
                "name": "non_hybrid_efficiency",
                "passed": False,
                "explanation": f"Non-hybrid vehicle claiming {mpg} MPG is unrealistic",
                "severity": "critical"
            }

        return {
            "name": "fuel_economy_range",
            "passed": True,
            "explanation": f"MPG {mpg} within realistic range",
            "severity": "info"
        }

    def _check_horsepower_constraint(self, hp: float, vehicle_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check horsepower against realistic constraints"""

        constraints = self.constraints["horsepower"]

        if hp < constraints["min_realistic_hp"] or hp > constraints["max_realistic_hp"]:
            return {
                "name": "horsepower_range",
                "passed": False,
                "explanation": f"Horsepower {hp} outside typical range ({constraints['min_realistic_hp']}-{constraints['max_realistic_hp']})",
                "severity": "caution"
            }

        return {
            "name": "horsepower_range",
            "passed": True,
            "explanation": f"Horsepower {hp} within realistic range",
            "severity": "info"
        }

    def _check_acceleration_constraint(self, zero_to_sixty: float, vehicle_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check acceleration against physics constraints"""

        constraints = self.constraints["acceleration"]

        if zero_to_sixty < constraints["min_realistic_0_60"] or zero_to_sixty > constraints["max_realistic_0_60"]:
            return {
                "name": "acceleration_range",
                "passed": False,
                "explanation": f"0-60 time {zero_to_sixty}s outside realistic range ({constraints['min_realistic_0_60']}-{constraints['max_realistic_0_60']})",
                "severity": "critical"
            }

        return {
            "name": "acceleration_range",
            "passed": True,
            "explanation": f"0-60 time {zero_to_sixty}s within realistic range",
            "severity": "info"
        }

    def _check_price_constraint(self, price: float, vehicle_context: Dict[str, Any]) -> Dict[str, Any]:
        """Check price against market constraints"""

        constraints = self.constraints["price"]

        if price < constraints["min_new_car"]:
            return {
                "name": "price_too_low",
                "passed": False,
                "explanation": f"Price ${price:,.0f} unusually low for new vehicle",
                "severity": "caution"
            }

        if price > constraints["max_reasonable"]:
            return {
                "name": "price_very_high",
                "passed": False,
                "explanation": f"Price ${price:,.0f} in supercar range - verify for regular vehicles",
                "severity": "caution"
            }

        return {
            "name": "price_range",
            "passed": True,
            "explanation": f"Price ${price:,.0f} within reasonable range",
            "severity": "info"
        }

    def _cross_validate_claims(
            self,
            claims_by_type: Dict[str, List[Dict[str, Any]]],
            vehicle_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Cross-validate related claims for consistency"""

        cross_checks = []

        # Check horsepower vs acceleration correlation
        hp_claims = claims_by_type.get("horsepower", [])
        accel_claims = claims_by_type.get("acceleration", [])

        if hp_claims and accel_claims:
            hp = hp_claims[0]["value"]
            zero_sixty = accel_claims[0]["value"]

            # Very rough correlation: more HP should mean faster acceleration
            # This is a simplified check - real correlation depends on weight, transmission, etc.
            expected_max_time = max(3.0, 15.0 - (hp / 50))  # Rough formula

            if zero_sixty > expected_max_time:
                cross_checks.append({
                    "name": "horsepower_acceleration_correlation",
                    "passed": False,
                    "explanation": f"{hp} HP with {zero_sixty}s 0-60 seems inconsistent",
                    "severity": "caution"
                })
            else:
                cross_checks.append({
                    "name": "horsepower_acceleration_correlation",
                    "passed": True,
                    "explanation": f"{hp} HP and {zero_sixty}s 0-60 seem consistent",
                    "severity": "info"
                })

        # Check fuel economy vs horsepower trade-off
        mpg_claims = claims_by_type.get("fuel_economy", [])
        if hp_claims and mpg_claims:
            hp = hp_claims[0]["value"]
            mpg = mpg_claims[0]["value"]

            # High horsepower usually means lower fuel economy (except hybrids)
            if hp > 300 and mpg > 35 and "hybrid" not in str(vehicle_context).lower():
                cross_checks.append({
                    "name": "power_efficiency_tradeoff",
                    "passed": False,
                    "explanation": f"High HP ({hp}) with high MPG ({mpg}) unusual for non-hybrid",
                    "severity": "caution"
                })

        return cross_checks