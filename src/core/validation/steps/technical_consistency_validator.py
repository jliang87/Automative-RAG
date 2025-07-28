"""
Technical Consistency Validator - COMPLETE UPDATED VERSION
Validates technical specifications consistency for automotive information using models
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.models import (
    ValidationStepResult, ValidationStatus, ValidationStepType,
    ValidationContext, ValidationWarning
)
from src.models.knowledge_models import (
    ValidationReferenceDatabase, FuelEconomyReference, EngineSpecification,
    SafetyRating, VehicleDimensions, PhysicsConstraint, ValidationConstraints,
    SourceType
)
from .steps_readiness_checker import MetaValidator, PreconditionResult

logger = logging.getLogger(__name__)


class TechnicalConsistencyValidator:
    """
    Validates consistency of technical specifications across sources using model-driven reference data
    """

    def __init__(self,
                 step_config: Dict[str, Any],
                 meta_validator: MetaValidator,
                 reference_database: Optional[ValidationReferenceDatabase] = None,
                 physics_constraints: Optional[ValidationConstraints] = None):
        self.step_config = step_config
        self.meta_validator = meta_validator
        self.step_type = ValidationStepType.TECHNICAL_CONSISTENCY

        # Use model-based reference data instead of hardcoded dict
        self.reference_database = reference_database or self._load_default_reference_database()
        self.physics_constraints = physics_constraints or self._load_default_physics_constraints()

        # Build lookup tables for performance
        self._build_reference_lookups()

    def _load_default_reference_database(self) -> ValidationReferenceDatabase:
        """Load default reference database using models instead of hardcoded data"""

        # Create model instances for EPA fuel economy data
        fuel_economy_data = [
            FuelEconomyReference(
                reference_id="epa_2023_toyota_camry",
                manufacturer="toyota",
                model="camry",
                year=2023,
                city_mpg=22,
                highway_mpg=32,
                combined_mpg=26,
                fuel_type="Regular Gasoline",
                source_type=SourceType.REGULATORY,
                source_url="https://www.fueleconomy.gov",
                verified_date=datetime.now()
            ),
            FuelEconomyReference(
                reference_id="epa_2023_honda_accord",
                manufacturer="honda",
                model="accord",
                year=2023,
                city_mpg=23,
                highway_mpg=34,
                combined_mpg=27,
                fuel_type="Regular Gasoline",
                source_type=SourceType.REGULATORY,
                source_url="https://www.fueleconomy.gov",
                verified_date=datetime.now()
            ),
            FuelEconomyReference(
                reference_id="epa_2023_ford_f150",
                manufacturer="ford",
                model="f150",
                year=2023,
                city_mpg=18,
                highway_mpg=24,
                combined_mpg=20,
                fuel_type="Regular Gasoline",
                source_type=SourceType.REGULATORY,
                source_url="https://www.fueleconomy.gov",
                verified_date=datetime.now()
            )
        ]

        # Create model instances for engine specifications
        engine_specifications = [
            EngineSpecification(
                spec_id="toyota_camry_2023_engine",
                manufacturer="toyota",
                model="camry",
                year=2023,
                horsepower=203,
                torque=184,
                displacement=2.5,
                engine_type="4-cylinder",
                fuel_type="Regular Gasoline",
                source_type=SourceType.OFFICIAL,
                verified=True
            ),
            EngineSpecification(
                spec_id="honda_accord_2023_engine",
                manufacturer="honda",
                model="accord",
                year=2023,
                horsepower=192,
                torque=192,
                displacement=1.5,
                engine_type="4-cylinder Turbo",
                fuel_type="Regular Gasoline",
                source_type=SourceType.OFFICIAL,
                verified=True
            )
        ]

        # Create model instances for safety ratings
        safety_ratings = [
            SafetyRating(
                rating_id="nhtsa_2023_toyota_camry",
                manufacturer="toyota",
                model="camry",
                year=2023,
                nhtsa_overall=5,
                nhtsa_frontal=5,
                nhtsa_side=5,
                nhtsa_rollover=4,
                iihs_top_safety_pick=True,
                source_type=SourceType.REGULATORY,
                test_date=datetime.now()
            ),
            SafetyRating(
                rating_id="nhtsa_2023_honda_accord",
                manufacturer="honda",
                model="accord",
                year=2023,
                nhtsa_overall=5,
                nhtsa_frontal=5,
                nhtsa_side=5,
                nhtsa_rollover=4,
                iihs_top_safety_pick=True,
                source_type=SourceType.REGULATORY,
                test_date=datetime.now()
            )
        ]

        # Create model instances for vehicle dimensions
        vehicle_dimensions = [
            VehicleDimensions(
                spec_id="toyota_camry_2023_dimensions",
                manufacturer="toyota",
                model="camry",
                year=2023,
                length=192.7,
                width=72.4,
                height=56.9,
                wheelbase=111.2,
                curb_weight=3340,
                cargo_volume=15.1,
                source_type=SourceType.OFFICIAL,
                verified=True
            )
        ]

        total_entries = len(fuel_economy_data) + len(engine_specifications) + len(safety_ratings) + len(vehicle_dimensions)

        return ValidationReferenceDatabase(
            fuel_economy_data=fuel_economy_data,
            engine_specifications=engine_specifications,
            safety_ratings=safety_ratings,
            vehicle_dimensions=vehicle_dimensions,
            last_updated=datetime.now(),
            total_entries=total_entries,
            coverage_stats={
                "fuel_economy": len(fuel_economy_data),
                "engine_specs": len(engine_specifications),
                "safety_ratings": len(safety_ratings),
                "dimensions": len(vehicle_dimensions)
            }
        )

    def _load_default_physics_constraints(self) -> ValidationConstraints:
        """Load physics constraints using models instead of hardcoded data"""

        # Fuel economy physics constraints
        fuel_economy_constraints = [
            PhysicsConstraint(
                constraint_id="highway_vs_city_mpg",
                constraint_type="fuel_economy",
                rule_expression="highway_mpg >= city_mpg",
                description="Highway MPG should be greater than or equal to city MPG",
                severity="warning",
                applies_to=["gasoline", "hybrid"],
                reasonable_ranges={
                    "city_mpg": {"min": 8, "max": 60},
                    "highway_mpg": {"min": 10, "max": 80},
                    "combined_mpg": {"min": 9, "max": 70}
                }
            ),
            PhysicsConstraint(
                constraint_id="combined_mpg_average",
                constraint_type="fuel_economy",
                rule_expression="city_mpg * 0.8 <= combined_mpg <= highway_mpg * 0.8 + city_mpg * 0.2",
                description="Combined MPG should be a weighted average of city and highway",
                severity="warning",
                applies_to=["all"],
                reasonable_ranges={}
            )
        ]

        # Engine physics constraints
        engine_constraints = [
            PhysicsConstraint(
                constraint_id="positive_power_values",
                constraint_type="engine",
                rule_expression="horsepower > 0 AND torque > 0",
                description="Engine power values must be positive",
                severity="error",
                applies_to=["all"],
                reasonable_ranges={
                    "horsepower": {"min": 100, "max": 800},
                    "torque": {"min": 100, "max": 700},
                    "displacement": {"min": 1.0, "max": 8.0}
                }
            )
        ]

        # Dimension constraints
        dimension_constraints = [
            PhysicsConstraint(
                constraint_id="reasonable_vehicle_dimensions",
                constraint_type="dimensions",
                rule_expression="length > 0 AND width > 0 AND height > 0 AND weight > 0",
                description="Vehicle dimensions must be positive values",
                severity="error",
                applies_to=["all"],
                reasonable_ranges={
                    "weight": {"min": 2000, "max": 8000},
                    "length": {"min": 140, "max": 250},
                    "width": {"min": 60, "max": 90},
                    "height": {"min": 50, "max": 85}
                }
            )
        ]

        return ValidationConstraints(
            fuel_economy_constraints=fuel_economy_constraints,
            engine_constraints=engine_constraints,
            dimension_constraints=dimension_constraints,
            performance_constraints=[],
            default_tolerance=0.1,
            strict_mode=False
        )

    def _build_reference_lookups(self):
        """Build lookup tables for efficient reference data access"""
        self.fuel_economy_lookup = {}
        self.engine_spec_lookup = {}
        self.safety_rating_lookup = {}
        self.dimension_lookup = {}

        # Build fuel economy lookup
        for ref in self.reference_database.fuel_economy_data:
            key = self._build_vehicle_key_from_ref(ref.manufacturer, ref.model, ref.year)
            self.fuel_economy_lookup[key] = ref

        # Build engine spec lookup
        for ref in self.reference_database.engine_specifications:
            key = self._build_vehicle_key_from_ref(ref.manufacturer, ref.model, ref.year)
            self.engine_spec_lookup[key] = ref

        # Build safety rating lookup
        for ref in self.reference_database.safety_ratings:
            key = self._build_vehicle_key_from_ref(ref.manufacturer, ref.model, ref.year)
            self.safety_rating_lookup[key] = ref

        # Build dimension lookup
        for ref in self.reference_database.vehicle_dimensions:
            key = self._build_vehicle_key_from_ref(ref.manufacturer, ref.model, ref.year)
            self.dimension_lookup[key] = ref

    def _build_vehicle_key_from_ref(self, manufacturer: str, model: str, year: int) -> str:
        """Build vehicle key from reference data"""
        return f"{year}_{manufacturer.lower()}_{model.lower()}"

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
        """Perform the actual technical consistency validation using models"""

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

        # Perform consistency checks using model-driven validation
        consistency_results = {
            "fuel_economy_consistency": self._check_fuel_economy_consistency(extracted_specs),
            "engine_specs_consistency": self._check_engine_specs_consistency(extracted_specs),
            "safety_ratings_consistency": self._check_safety_ratings_consistency(extracted_specs),
            "dimension_consistency": self._check_dimension_consistency(extracted_specs),
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
            "specifications_found": self._summarize_specifications(extracted_specs),
            "reference_database_entries": self.reference_database.total_entries,
            "physics_constraints_applied": len(self.physics_constraints.fuel_economy_constraints +
                                             self.physics_constraints.engine_constraints +
                                             self.physics_constraints.dimension_constraints)
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

    def _validate_against_reference_data(self, extracted_specs: List[Dict], context: ValidationContext) -> Dict[str, Any]:
        """Validate extracted specifications against model-based reference databases"""

        issues = []
        matches_found = 0

        # Try to match vehicle from context
        vehicle_key = self._build_vehicle_key(context)

        for spec in extracted_specs:
            source_id = spec["source_id"]

            # Check fuel economy against EPA data using models
            if "fuel_economy" in spec["extracted_specs"]:
                epa_ref = self._find_fuel_economy_reference(context, vehicle_key)
                if epa_ref:
                    matches_found += 1
                    fuel_issues = self._compare_with_epa_reference(
                        spec["extracted_specs"]["fuel_economy"], epa_ref, source_id
                    )
                    issues.extend(fuel_issues)

            # Check engine specs against manufacturer data using models
            if "engine" in spec["extracted_specs"]:
                engine_ref = self._find_engine_reference(context, vehicle_key)
                if engine_ref:
                    matches_found += 1
                    engine_issues = self._compare_with_engine_reference(
                        spec["extracted_specs"]["engine"], engine_ref, source_id
                    )
                    issues.extend(engine_issues)

            # Check safety ratings using models
            if "safety" in spec["extracted_specs"]:
                safety_ref = self._find_safety_reference(context, vehicle_key)
                if safety_ref:
                    matches_found += 1
                    safety_issues = self._compare_with_safety_reference(
                        spec["extracted_specs"]["safety"], safety_ref, source_id
                    )
                    issues.extend(safety_issues)

        return {
            "status": "inconsistent" if issues else "consistent",
            "issues": issues,
            "reference_matches": matches_found,
            "total_reference_entries_checked": self.reference_database.total_entries
        }

    def _find_fuel_economy_reference(self, context: ValidationContext, vehicle_key: Optional[str]) -> Optional[FuelEconomyReference]:
        """Find matching fuel economy reference using model-based lookup"""
        if vehicle_key and vehicle_key in self.fuel_economy_lookup:
            return self.fuel_economy_lookup[vehicle_key]

        # Fallback to manual search if key lookup fails
        for ref in self.reference_database.fuel_economy_data:
            if self._matches_vehicle_context(ref.manufacturer, ref.model, ref.year, context):
                return ref
        return None

    def _find_engine_reference(self, context: ValidationContext, vehicle_key: Optional[str]) -> Optional[EngineSpecification]:
        """Find matching engine specification using model-based lookup"""
        if vehicle_key and vehicle_key in self.engine_spec_lookup:
            return self.engine_spec_lookup[vehicle_key]

        # Fallback to manual search
        for ref in self.reference_database.engine_specifications:
            if self._matches_vehicle_context(ref.manufacturer, ref.model, ref.year, context):
                return ref
        return None

    def _find_safety_reference(self, context: ValidationContext, vehicle_key: Optional[str]) -> Optional[SafetyRating]:
        """Find matching safety rating using model-based lookup"""
        if vehicle_key and vehicle_key in self.safety_rating_lookup:
            return self.safety_rating_lookup[vehicle_key]

        # Fallback to manual search
        for ref in self.reference_database.safety_ratings:
            if self._matches_vehicle_context(ref.manufacturer, ref.model, ref.year, context):
                return ref
        return None

    def _matches_vehicle_context(self, ref_manufacturer: str, ref_model: str, ref_year: int, context: ValidationContext) -> bool:
        """Check if reference data matches vehicle context"""
        manufacturer_match = (not context.manufacturer or
                            ref_manufacturer.lower() == context.manufacturer.lower())
        model_match = (not context.model or
                      ref_model.lower() == context.model.lower())
        year_match = (not context.year or ref_year == context.year)

        return manufacturer_match and model_match and year_match

    def _compare_with_epa_reference(self, extracted_fuel: Dict, epa_ref: FuelEconomyReference, source_id: str) -> List[Dict]:
        """Compare extracted fuel economy with EPA reference using models"""
        issues = []
        tolerance = 2  # Allow 2 MPG tolerance

        # Compare city MPG
        if "city_mpg" in extracted_fuel and epa_ref.city_mpg:
            if abs(extracted_fuel["city_mpg"] - epa_ref.city_mpg) > tolerance:
                issues.append({
                    "message": f"EPA data mismatch for city MPG",
                    "explanation": f"Source: {extracted_fuel['city_mpg']} MPG, EPA: {epa_ref.city_mpg} MPG",
                    "suggestion": "Verify against official EPA database",
                    "severity": "medium",
                    "source": source_id,
                    "reference_id": epa_ref.reference_id
                })

        # Compare highway MPG
        if "highway_mpg" in extracted_fuel and epa_ref.highway_mpg:
            if abs(extracted_fuel["highway_mpg"] - epa_ref.highway_mpg) > tolerance:
                issues.append({
                    "message": f"EPA data mismatch for highway MPG",
                    "explanation": f"Source: {extracted_fuel['highway_mpg']} MPG, EPA: {epa_ref.highway_mpg} MPG",
                    "suggestion": "Verify against official EPA database",
                    "severity": "medium",
                    "source": source_id,
                    "reference_id": epa_ref.reference_id
                })

        # Compare combined MPG
        if "combined_mpg" in extracted_fuel and epa_ref.combined_mpg:
            if abs(extracted_fuel["combined_mpg"] - epa_ref.combined_mpg) > tolerance:
                issues.append({
                    "message": f"EPA data mismatch for combined MPG",
                    "explanation": f"Source: {extracted_fuel['combined_mpg']} MPG, EPA: {epa_ref.combined_mpg} MPG",
                    "suggestion": "Verify against official EPA database",
                    "severity": "medium",
                    "source": source_id,
                    "reference_id": epa_ref.reference_id
                })

        return issues

    def _compare_with_engine_reference(self, extracted_engine: Dict, engine_ref: EngineSpecification, source_id: str) -> List[Dict]:
        """Compare extracted engine specs with reference using models"""
        issues = []

        # Compare horsepower (allow 5% tolerance)
        if "horsepower" in extracted_engine and engine_ref.horsepower:
            tolerance = engine_ref.horsepower * 0.05
            if abs(extracted_engine["horsepower"] - engine_ref.horsepower) > tolerance:
                issues.append({
                    "message": "Horsepower mismatch with reference data",
                    "explanation": f"Source: {extracted_engine['horsepower']} HP, Reference: {engine_ref.horsepower} HP",
                    "suggestion": "Verify against official manufacturer specifications",
                    "severity": "medium",
                    "source": source_id,
                    "reference_id": engine_ref.spec_id
                })

        # Compare torque (allow 5% tolerance)
        if "torque" in extracted_engine and engine_ref.torque:
            tolerance = engine_ref.torque * 0.05
            if abs(extracted_engine["torque"] - engine_ref.torque) > tolerance:
                issues.append({
                    "message": "Torque mismatch with reference data",
                    "explanation": f"Source: {extracted_engine['torque']} lb-ft, Reference: {engine_ref.torque} lb-ft",
                    "suggestion": "Verify against official manufacturer specifications",
                    "severity": "medium",
                    "source": source_id,
                    "reference_id": engine_ref.spec_id
                })

        return issues

    def _compare_with_safety_reference(self, extracted_safety: Dict, safety_ref: SafetyRating, source_id: str) -> List[Dict]:
        """Compare extracted safety ratings with reference using models"""
        issues = []

        # Compare NHTSA overall rating
        if "nhtsa_overall" in extracted_safety and safety_ref.nhtsa_overall:
            if extracted_safety["nhtsa_overall"] != safety_ref.nhtsa_overall:
                issues.append({
                    "message": "NHTSA overall rating mismatch",
                    "explanation": f"Source: {extracted_safety['nhtsa_overall']} stars, Reference: {safety_ref.nhtsa_overall} stars",
                    "suggestion": "Verify against official NHTSA database",
                    "severity": "high",
                    "source": source_id,
                    "reference_id": safety_ref.rating_id
                })

        # Compare IIHS Top Safety Pick status
        if "iihs_top_safety_pick" in extracted_safety and safety_ref.iihs_top_safety_pick is not None:
            if extracted_safety["iihs_top_safety_pick"] != safety_ref.iihs_top_safety_pick:
                issues.append({
                    "message": "IIHS Top Safety Pick status mismatch",
                    "explanation": f"Source: {extracted_safety['iihs_top_safety_pick']}, Reference: {safety_ref.iihs_top_safety_pick}",
                    "suggestion": "Verify against official IIHS database",
                    "severity": "medium",
                    "source": source_id,
                    "reference_id": safety_ref.rating_id
                })

        return issues

    def _validate_physics_constraints(self, extracted_specs: List[Dict]) -> Dict[str, Any]:
        """Validate extracted specifications against physics constraints using models"""

        issues = []

        for spec in extracted_specs:
            source_id = spec["source_id"]

            # Validate fuel economy physics using model constraints
            if "fuel_economy" in spec["extracted_specs"]:
                fuel_data = spec["extracted_specs"]["fuel_economy"]
                fuel_issues = self._validate_fuel_economy_physics_with_models(fuel_data, source_id)
                issues.extend(fuel_issues)

            # Validate engine specs physics using model constraints
            if "engine" in spec["extracted_specs"]:
                engine_data = spec["extracted_specs"]["engine"]
                engine_issues = self._validate_engine_physics_with_models(engine_data, source_id)
                issues.extend(engine_issues)

            # Validate dimensions physics using model constraints
            if "dimensions" in spec["extracted_specs"]:
                dimension_data = spec["extracted_specs"]["dimensions"]
                dimension_issues = self._validate_dimension_physics_with_models(dimension_data, source_id)
                issues.extend(dimension_issues)

        return {
            "status": "inconsistent" if issues else "consistent",
            "issues": issues,
            "physics_checks_performed": len(extracted_specs),
            "constraints_applied": len(self.physics_constraints.fuel_economy_constraints +
                                    self.physics_constraints.engine_constraints +
                                    self.physics_constraints.dimension_constraints)
        }

    def _validate_fuel_economy_physics_with_models(self, fuel_data: Dict, source_id: str) -> List[Dict]:
        """Validate fuel economy data against model-based physics constraints"""

        issues = []

        for constraint in self.physics_constraints.fuel_economy_constraints:
            # Check reasonable ranges
            for mpg_type, value in fuel_data.items():
                if mpg_type in constraint.reasonable_ranges:
                    range_constraint = constraint.reasonable_ranges[mpg_type]
                    if not (range_constraint["min"] <= value <= range_constraint["max"]):
                        issues.append({
                            "message": f"Unrealistic {mpg_type}: {value}",
                            "explanation": f"Value outside reasonable range ({range_constraint['min']}-{range_constraint['max']})",
                            "suggestion": "Verify this specification",
                            "severity": "high" if constraint.severity == "error" else "medium",
                            "source": source_id,
                            "constraint_id": constraint.constraint_id
                        })

            # Check physics rules
            if constraint.rule_expression == "highway_mpg >= city_mpg":
                if "city_mpg" in fuel_data and "highway_mpg" in fuel_data:
                    if fuel_data["highway_mpg"] < fuel_data["city_mpg"]:
                        issues.append({
                            "message": "Highway MPG lower than city MPG",
                            "explanation": f"Highway: {fuel_data['highway_mpg']}, City: {fuel_data['city_mpg']}",
                            "suggestion": constraint.description,
                            "severity": "medium" if constraint.severity == "warning" else "high",
                            "source": source_id,
                            "constraint_id": constraint.constraint_id
                        })

        return issues

    def _validate_engine_physics_with_models(self, engine_data: Dict, source_id: str) -> List[Dict]:
        """Validate engine specifications against model-based physics constraints"""

        issues = []

        for constraint in self.physics_constraints.engine_constraints:
            # Check reasonable ranges
            for spec_type, value in engine_data.items():
                if spec_type in constraint.reasonable_ranges:
                    range_constraint = constraint.reasonable_ranges[spec_type]
                    if not (range_constraint["min"] <= value <= range_constraint["max"]):
                        issues.append({
                            "message": f"Unrealistic {spec_type}: {value}",
                            "explanation": f"Value outside reasonable range ({range_constraint['min']}-{range_constraint['max']})",
                            "suggestion": "Verify this specification",
                            "severity": "high" if constraint.severity == "error" else "medium",
                            "source": source_id,
                            "constraint_id": constraint.constraint_id
                        })

        return issues

    def _validate_dimension_physics_with_models(self, dimension_data: Dict, source_id: str) -> List[Dict]:
        """Validate dimension data against model-based physics constraints"""

        issues = []

        for constraint in self.physics_constraints.dimension_constraints:
            # Check reasonable ranges
            for dim_type, value in dimension_data.items():
                if dim_type in constraint.reasonable_ranges:
                    range_constraint = constraint.reasonable_ranges[dim_type]
                    if not (range_constraint["min"] <= value <= range_constraint["max"]):
                        issues.append({
                            "message": f"Unrealistic {dim_type}: {value}",
                            "explanation": f"Value outside reasonable range ({range_constraint['min']}-{range_constraint['max']})",
                            "suggestion": "Verify this specification",
                            "severity": "high" if constraint.severity == "error" else "medium",
                            "source": source_id,
                            "constraint_id": constraint.constraint_id
                        })

        return issues

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