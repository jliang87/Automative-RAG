"""
Completeness Analysis Validator
Analyzes context completeness for automotive queries
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

from src.models import (
    ValidationStepResult, ValidationStatus, ValidationStepType,
    ValidationContext, ValidationWarning
)
from .steps_readiness_checker import MetaValidator, PreconditionResult

logger = logging.getLogger(__name__)


class CompletenessValidator:
    """
    Analyzes completeness of information for automotive queries
    """

    def __init__(self, step_config: Dict[str, Any], meta_validator: MetaValidator):
        self.step_config = step_config
        self.meta_validator = meta_validator
        self.step_type = ValidationStepType.COMPLETENESS

        # Define completeness requirements for different query modes
        self.completeness_requirements = self._initialize_completeness_requirements()

    def _initialize_completeness_requirements(self) -> Dict[str, Dict[str, Any]]:
        """Initialize completeness requirements for different query modes"""
        return {
            "facts": {
                "required_context": ["manufacturer", "model"],
                "optional_context": ["year", "trim", "market"],
                "required_information_types": ["specifications", "official_data"],
                "minimum_sources": 2,
                "preferred_source_types": ["official", "regulatory", "professional"]
            },
            "features": {
                "required_context": ["manufacturer", "model"],
                "optional_context": ["year", "trim"],
                "required_information_types": ["feature_descriptions", "comparative_data"],
                "minimum_sources": 3,
                "preferred_source_types": ["professional", "official", "user_generated"]
            },
            "tradeoffs": {
                "required_context": ["manufacturer", "model"],
                "optional_context": ["year", "use_case"],
                "required_information_types": ["advantages", "disadvantages", "comparative_analysis"],
                "minimum_sources": 3,
                "preferred_source_types": ["professional", "user_generated", "official"]
            },
            "scenarios": {
                "required_context": ["use_case", "requirements"],
                "optional_context": ["budget", "location", "family_size"],
                "required_information_types": ["use_case_analysis", "vehicle_suitability"],
                "minimum_sources": 2,
                "preferred_source_types": ["professional", "user_generated"]
            },
            "debate": {
                "required_context": ["topic", "perspectives"],
                "optional_context": ["timeframe", "market"],
                "required_information_types": ["expert_opinions", "evidence", "counterarguments"],
                "minimum_sources": 4,
                "preferred_source_types": ["professional", "academic", "official"]
            },
            "quotes": {
                "required_context": ["vehicle_context"],
                "optional_context": ["user_demographics", "use_patterns"],
                "required_information_types": ["user_experiences", "real_world_feedback"],
                "minimum_sources": 3,
                "preferred_source_types": ["user_generated", "professional"]
            }
        }

    async def execute(self, context: ValidationContext) -> ValidationStepResult:
        """Execute completeness analysis"""

        start_time = datetime.now()

        # Check preconditions
        precondition_result = await self.meta_validator.check_preconditions(
            self.step_type, context, self.step_config
        )

        if precondition_result.status != "READY":
            return self._create_unverifiable_result(start_time, precondition_result)

        try:
            # Perform completeness analysis
            result = await self._perform_validation(context)
            result.completed_at = datetime.now()
            result.duration_ms = int((result.completed_at - start_time).total_seconds() * 1000)

            return result

        except Exception as e:
            logger.error(f"Completeness analysis failed: {str(e)}")
            return self._create_error_result(start_time, str(e))

    async def _perform_validation(self, context: ValidationContext) -> ValidationStepResult:
        """Perform the actual completeness analysis"""

        documents = context.documents
        query_mode = context.query_mode
        warnings = []

        # Get requirements for this query mode
        requirements = self.completeness_requirements.get(query_mode, self.completeness_requirements["facts"])

        # Analyze different aspects of completeness
        completeness_analysis = {
            "context_completeness": self._analyze_context_completeness(context, requirements),
            "source_completeness": self._analyze_source_completeness(documents, requirements),
            "information_type_completeness": self._analyze_information_type_completeness(documents, requirements),
            "coverage_completeness": self._analyze_coverage_completeness(documents, context, requirements)
        }

        # Calculate overall completeness scores
        context_score = completeness_analysis["context_completeness"]["score"]
        source_score = completeness_analysis["source_completeness"]["score"]
        info_type_score = completeness_analysis["information_type_completeness"]["score"]
        coverage_score = completeness_analysis["coverage_completeness"]["score"]

        # Weight the scores based on importance
        weights = {"context": 0.3, "source": 0.25, "info_type": 0.25, "coverage": 0.2}
        overall_completeness = (
            context_score * weights["context"] +
            source_score * weights["source"] +
            info_type_score * weights["info_type"] +
            coverage_score * weights["coverage"]
        )

        # Generate warnings for completeness gaps
        for analysis_type, analysis_result in completeness_analysis.items():
            if analysis_result["score"] < 0.7:  # Below 70% completeness
                for gap in analysis_result.get("gaps", []):
                    severity = "critical" if gap.get("critical", False) else "caution"
                    warnings.append(ValidationWarning(
                        category=analysis_type,
                        severity=severity,
                        message=gap["message"],
                        explanation=gap.get("explanation", ""),
                        suggestion=gap.get("suggestion", "Add more comprehensive information")
                    ))

        # Determine validation status and confidence impact
        if overall_completeness >= 0.85:
            status = ValidationStatus.PASSED
            confidence_impact = 8.0 + (overall_completeness - 0.85) * 20  # 8-11 point boost
        elif overall_completeness >= 0.7:
            status = ValidationStatus.WARNING
            confidence_impact = 2.0 + (overall_completeness - 0.7) * 20  # 2-5 point boost
        else:
            status = ValidationStatus.WARNING
            confidence_impact = -2.0 + (overall_completeness * 5)  # Up to -2 point penalty

            warnings.append(ValidationWarning(
                category="overall_completeness",
                severity="critical",
                message="Information completeness is insufficient",
                explanation=f"Overall completeness: {overall_completeness:.1%}",
                suggestion="Add more comprehensive sources and context information"
            ))

        # Build summary
        summary = (f"Completeness analysis for {query_mode} mode. "
                  f"Overall score: {overall_completeness:.1%}. "
                  f"Context: {context_score:.1%}, Sources: {source_score:.1%}, "
                  f"Coverage: {coverage_score:.1%}")

        # Build detailed results
        details = {
            "query_mode": query_mode,
            "overall_completeness_score": overall_completeness,
            "component_scores": {
                "context_completeness": context_score,
                "source_completeness": source_score,
                "information_type_completeness": info_type_score,
                "coverage_completeness": coverage_score
            },
            "completeness_analysis": completeness_analysis,
            "requirements_used": requirements,
            "missing_critical_information": self._identify_critical_gaps(completeness_analysis),
            "improvement_suggestions": self._generate_improvement_suggestions(completeness_analysis, query_mode)
        }

        return ValidationStepResult(
            step_id=f"completeness_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Completeness Analysis",
            status=status,
            confidence_impact=confidence_impact,
            summary=summary,
            details=details,
            started_at=start_time,
            warnings=warnings,
            sources_used=[f"document_{i}" for i in range(len(documents))]
        )

    def _analyze_context_completeness(self, context: ValidationContext, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze completeness of query context"""

        required_context = requirements.get("required_context", [])
        optional_context = requirements.get("optional_context", [])

        # Check required context elements
        context_values = {
            "manufacturer": context.manufacturer,
            "model": context.model,
            "year": context.year,
            "trim": context.trim,
            "market": context.market,
            "use_case": self._extract_use_case_from_query(context.query_text),
            "topic": self._extract_topic_from_query(context.query_text),
            "vehicle_context": bool(context.manufacturer or context.model)
        }

        missing_required = []
        missing_optional = []

        for req_item in required_context:
            if not context_values.get(req_item):
                missing_required.append(req_item)

        for opt_item in optional_context:
            if not context_values.get(opt_item):
                missing_optional.append(opt_item)

        # Calculate score
        required_present = len(required_context) - len(missing_required)
        optional_present = len(optional_context) - len(missing_optional)

        if len(required_context) > 0:
            required_score = required_present / len(required_context)
        else:
            required_score = 1.0

        if len(optional_context) > 0:
            optional_score = optional_present / len(optional_context)
        else:
            optional_score = 1.0

        # Weight required more heavily
        overall_score = (required_score * 0.8) + (optional_score * 0.2)

        gaps = []
        if missing_required:
            gaps.append({
                "message": f"Missing required context: {', '.join(missing_required)}",
                "explanation": "These context elements are essential for this query type",
                "suggestion": "Provide more specific information in the query",
                "critical": True
            })

        if missing_optional and len(missing_optional) > len(optional_context) / 2:
            gaps.append({
                "message": f"Limited optional context: {', '.join(missing_optional)}",
                "explanation": "Additional context would improve analysis quality",
                "suggestion": "Consider providing more specific details",
                "critical": False
            })

        return {
            "score": overall_score,
            "required_present": required_present,
            "optional_present": optional_present,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "gaps": gaps
        }

    def _analyze_source_completeness(self, documents: List[Dict], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze completeness of sources"""

        minimum_sources = requirements.get("minimum_sources", 2)
        preferred_source_types = requirements.get("preferred_source_types", [])

        # Analyze source types present
        source_types_present = set()
        for doc in documents:
            metadata = doc.get("metadata", {})
            source_type = self._determine_source_type(metadata)
            source_types_present.add(source_type)

        gaps = []

        # Check minimum source count
        if len(documents) < minimum_sources:
            gaps.append({
                "message": f"Insufficient sources: {len(documents)} of {minimum_sources} minimum",
                "explanation": "More sources would improve validation reliability",
                "suggestion": "Add additional authoritative sources",
                "critical": True
            })

        # Check source type diversity
        missing_preferred_types = []
        for preferred_type in preferred_source_types:
            if preferred_type not in source_types_present:
                missing_preferred_types.append(preferred_type)

        if len(missing_preferred_types) > len(preferred_source_types) / 2:
            gaps.append({
                "message": f"Limited source type diversity",
                "explanation": f"Missing preferred types: {', '.join(missing_preferred_types)}",
                "suggestion": "Add sources from different categories (official, professional, user reviews)",
                "critical": False
            })

        # Calculate score
        source_count_score = min(1.0, len(documents) / minimum_sources)
        source_diversity_score = len(source_types_present) / len(preferred_source_types) if preferred_source_types else 1.0

        overall_score = (source_count_score * 0.6) + (source_diversity_score * 0.4)

        return {
            "score": overall_score,
            "source_count": len(documents),
            "minimum_required": minimum_sources,
            "source_types_present": list(source_types_present),
            "preferred_types": preferred_source_types,
            "missing_preferred_types": missing_preferred_types,
            "gaps": gaps
        }

    def _analyze_information_type_completeness(self, documents: List[Dict], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze completeness of information types"""

        required_info_types = requirements.get("required_information_types", [])

        # Analyze what types of information are present
        info_types_present = set()
        for doc in documents:
            content = doc.get("content", "").lower()
            metadata = doc.get("metadata", {})

            # Check for different information types
            if self._contains_specifications(content, metadata):
                info_types_present.add("specifications")
            if self._contains_official_data(content, metadata):
                info_types_present.add("official_data")
            if self._contains_feature_descriptions(content):
                info_types_present.add("feature_descriptions")
            if self._contains_comparative_data(content):
                info_types_present.add("comparative_data")
            if self._contains_advantages_disadvantages(content):
                info_types_present.add("advantages")
                info_types_present.add("disadvantages")
            if self._contains_comparative_analysis(content):
                info_types_present.add("comparative_analysis")
            if self._contains_use_case_analysis(content):
                info_types_present.add("use_case_analysis")
            if self._contains_vehicle_suitability(content):
                info_types_present.add("vehicle_suitability")
            if self._contains_expert_opinions(content):
                info_types_present.add("expert_opinions")
            if self._contains_evidence(content):
                info_types_present.add("evidence")
            if self._contains_counterarguments(content):
                info_types_present.add("counterarguments")
            if self._contains_user_experiences(content):
                info_types_present.add("user_experiences")
            if self._contains_real_world_feedback(content):
                info_types_present.add("real_world_feedback")

        # Check for missing required information types
        missing_info_types = []
        for req_type in required_info_types:
            if req_type not in info_types_present:
                missing_info_types.append(req_type)

        gaps = []
        if missing_info_types:
            gaps.append({
                "message": f"Missing required information types: {', '.join(missing_info_types)}",
                "explanation": "These information types are essential for comprehensive analysis",
                "suggestion": "Add sources that contain the missing information types",
                "critical": True
            })

        # Calculate score
        if required_info_types:
            score = (len(required_info_types) - len(missing_info_types)) / len(required_info_types)
        else:
            score = 1.0

        return {
            "score": score,
            "required_info_types": required_info_types,
            "info_types_present": list(info_types_present),
            "missing_info_types": missing_info_types,
            "gaps": gaps
        }

    def _analyze_coverage_completeness(self, documents: List[Dict], context: ValidationContext, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze completeness of coverage for the specific query"""

        # Extract key topics/aspects from the query
        query_aspects = self._extract_query_aspects(context.query_text, context.query_mode)

        # Check what aspects are covered by the documents
        covered_aspects = set()
        aspect_coverage = {}

        for aspect in query_aspects:
            coverage_count = 0
            for doc in documents:
                content = doc.get("content", "").lower()
                if self._aspect_covered_in_content(aspect, content):
                    coverage_count += 1

            if coverage_count > 0:
                covered_aspects.add(aspect)
                aspect_coverage[aspect] = coverage_count

        # Calculate coverage completeness
        if query_aspects:
            coverage_score = len(covered_aspects) / len(query_aspects)
        else:
            coverage_score = 1.0

        # Check for depth of coverage
        shallow_coverage = []
        for aspect in covered_aspects:
            if aspect_coverage[aspect] < 2:  # Less than 2 sources
                shallow_coverage.append(aspect)

        gaps = []
        missing_aspects = [asp for asp in query_aspects if asp not in covered_aspects]

        if missing_aspects:
            gaps.append({
                "message": f"Query aspects not covered: {', '.join(missing_aspects)}",
                "explanation": "Some aspects of the query are not addressed by available sources",
                "suggestion": "Find sources that specifically address these aspects",
                "critical": True
            })

        if shallow_coverage:
            gaps.append({
                "message": f"Shallow coverage for: {', '.join(shallow_coverage)}",
                "explanation": "These aspects are covered by only one source",
                "suggestion": "Add additional sources for better coverage depth",
                "critical": False
            })

        return {
            "score": coverage_score,
            "query_aspects": query_aspects,
            "covered_aspects": list(covered_aspects),
            "missing_aspects": missing_aspects,
            "shallow_coverage": shallow_coverage,
            "aspect_coverage": aspect_coverage,
            "gaps": gaps
        }

    def _extract_use_case_from_query(self, query_text: str) -> Optional[str]:
        """Extract use case information from query text"""

        use_case_indicators = [
            "commuting", "family", "towing", "off-road", "city driving",
            "highway", "long distance", "daily driver", "weekend",
            "work", "business", "cargo", "passengers"
        ]

        query_lower = query_text.lower()
        for indicator in use_case_indicators:
            if indicator in query_lower:
                return indicator

        return None

    def _extract_topic_from_query(self, query_text: str) -> Optional[str]:
        """Extract main topic from query text"""

        # Simple topic extraction based on keywords
        topic_keywords = [
            "fuel economy", "safety", "reliability", "performance",
            "electric", "hybrid", "gas", "diesel", "maintenance",
            "cost", "price", "value", "technology", "features"
        ]

        query_lower = query_text.lower()
        for keyword in topic_keywords:
            if keyword in query_lower:
                return keyword

        return "automotive"

    def _determine_source_type(self, metadata: Dict[str, Any]) -> str:
        """Determine the type of source"""

        url = metadata.get("url", "").lower()
        source_platform = metadata.get("sourcePlatform", "").lower()

        if any(domain in url for domain in ['.gov', 'epa.', 'nhtsa.']):
            return "regulatory"
        elif any(domain in url for domain in ['toyota.com', 'honda.com', 'ford.com']):
            return "official"
        elif any(domain in url for domain in ['edmunds.com', 'motortrend.com', 'caranddriver.com']):
            return "professional"
        elif any(platform in source_platform for platform in ['reddit', 'forum', 'review']):
            return "user_generated"
        elif any(domain in url for domain in ['.edu', 'sae.org', 'ieee.org']):
            return "academic"
        else:
            return "other"

    def _contains_specifications(self, content: str, metadata: Dict) -> bool:
        """Check if content contains technical specifications"""

        spec_indicators = [
            "mpg", "horsepower", "torque", "engine", "transmission",
            "weight", "dimensions", "capacity", "displacement"
        ]

        return any(indicator in content for indicator in spec_indicators)

    def _contains_official_data(self, content: str, metadata: Dict) -> bool:
        """Check if content contains official data"""

        url = metadata.get("url", "").lower()
        official_indicators = [".gov", "epa", "nhtsa", "manufacturer"]

        return any(indicator in url for indicator in official_indicators) or \
               any(indicator in content for indicator in ["epa", "nhtsa", "official"])

    def _contains_feature_descriptions(self, content: str) -> bool:
        """Check if content contains feature descriptions"""

        feature_indicators = [
            "features", "equipment", "options", "technology", "system",
            "infotainment", "safety features", "comfort", "convenience"
        ]

        return any(indicator in content for indicator in feature_indicators)

    def _contains_comparative_data(self, content: str) -> bool:
        """Check if content contains comparative information"""

        comparison_indicators = [
            "vs", "versus", "compared to", "better than", "worse than",
            "comparison", "against", "relative to", "than"
        ]

        return any(indicator in content for indicator in comparison_indicators)

    def _contains_advantages_disadvantages(self, content: str) -> bool:
        """Check if content discusses advantages and disadvantages"""

        advantage_indicators = [
            "pros", "advantages", "benefits", "strengths", "positive",
            "cons", "disadvantages", "drawbacks", "weaknesses", "negative"
        ]

        return any(indicator in content for indicator in advantage_indicators)

    def _contains_comparative_analysis(self, content: str) -> bool:
        """Check if content contains comparative analysis"""

        analysis_indicators = [
            "analysis", "compare", "evaluation", "assessment", "review",
            "better choice", "recommendation"
        ]

        return any(indicator in content for indicator in analysis_indicators)

    def _contains_use_case_analysis(self, content: str) -> bool:
        """Check if content contains use case analysis"""

        use_case_indicators = [
            "use case", "scenario", "situation", "purpose", "intended for",
            "best for", "ideal for", "suitable for"
        ]

        return any(indicator in content for indicator in use_case_indicators)

    def _contains_vehicle_suitability(self, content: str) -> bool:
        """Check if content discusses vehicle suitability"""

        suitability_indicators = [
            "suitable", "appropriate", "fit", "match", "right for",
            "works well", "performance in"
        ]

        return any(indicator in content for indicator in suitability_indicators)

    def _contains_expert_opinions(self, content: str) -> bool:
        """Check if content contains expert opinions"""

        expert_indicators = [
            "expert", "analyst", "journalist", "reviewer", "professional",
            "opinion", "thinks", "believes", "according to"
        ]

        return any(indicator in content for indicator in expert_indicators)

    def _contains_evidence(self, content: str) -> bool:
        """Check if content contains supporting evidence"""

        evidence_indicators = [
            "data", "statistics", "study", "research", "test", "results",
            "evidence", "proof", "demonstrates", "shows"
        ]

        return any(indicator in content for indicator in evidence_indicators)

    def _contains_counterarguments(self, content: str) -> bool:
        """Check if content contains counterarguments"""

        counter_indicators = [
            "however", "but", "although", "despite", "on the other hand",
            "alternatively", "critics", "disagree"
        ]

        return any(indicator in content for indicator in counter_indicators)

    def _contains_user_experiences(self, content: str) -> bool:
        """Check if content contains user experiences"""

        experience_indicators = [
            "experience", "owner", "drove", "owned", "user", "customer",
            "my", "i have", "personal", "real world"
        ]

        return any(indicator in content for indicator in experience_indicators)

    def _contains_real_world_feedback(self, content: str) -> bool:
        """Check if content contains real-world feedback"""

        feedback_indicators = [
            "feedback", "review", "rating", "satisfaction", "problems",
            "issues", "reliability", "long term", "after"
        ]

        return any(indicator in content for indicator in feedback_indicators)

    def _extract_query_aspects(self, query_text: str, query_mode: str) -> List[str]:
        """Extract key aspects that should be covered based on the query"""

        aspects = []
        query_lower = query_text.lower()

        # Common automotive aspects
        automotive_aspects = [
            "fuel economy", "performance", "safety", "reliability",
            "features", "price", "maintenance", "comfort", "technology"
        ]

        for aspect in automotive_aspects:
            if aspect in query_lower:
                aspects.append(aspect)

        # Add mode-specific aspects
        if query_mode == "tradeoffs":
            aspects.extend(["advantages", "disadvantages"])
        elif query_mode == "scenarios":
            aspects.extend(["use cases", "suitability"])
        elif query_mode == "debate":
            aspects.extend(["expert opinions", "evidence"])
        elif query_mode == "quotes":
            aspects.extend(["user experiences", "real world feedback"])

        # If no specific aspects found, add general ones
        if not aspects:
            aspects = ["general information", "basic specifications"]

        return aspects

    def _aspect_covered_in_content(self, aspect: str, content: str) -> bool:
        """Check if a specific aspect is covered in the content"""

        aspect_keywords = {
            "fuel economy": ["mpg", "fuel", "economy", "gas mileage", "efficiency"],
            "performance": ["horsepower", "torque", "acceleration", "speed", "performance"],
            "safety": ["safety", "crash", "airbag", "nhtsa", "iihs"],
            "reliability": ["reliability", "dependable", "problems", "issues", "durable"],
            "features": ["features", "equipment", "technology", "infotainment"],
            "price": ["price", "cost", "expensive", "affordable", "value"],
            "maintenance": ["maintenance", "service", "repair", "oil change"],
            "comfort": ["comfort", "ride", "quiet", "spacious", "interior"],
            "technology": ["technology", "infotainment", "connectivity", "apps"],
            "advantages": ["pros", "advantages", "benefits", "positive"],
            "disadvantages": ["cons", "disadvantages", "drawbacks", "negative"],
            "use cases": ["use", "purpose", "ideal for", "best for"],
            "suitability": ["suitable", "appropriate", "fit", "match"],
            "expert opinions": ["expert", "analyst", "review", "professional"],
            "evidence": ["data", "study", "research", "test", "evidence"],
            "user experiences": ["owner", "experience", "user", "personal"],
            "real world feedback": ["feedback", "real world", "long term", "actual"]
        }

        keywords = aspect_keywords.get(aspect, [aspect])
        return any(keyword in content for keyword in keywords)

    def _identify_critical_gaps(self, completeness_analysis: Dict[str, Any]) -> List[str]:
        """Identify critical information gaps"""

        critical_gaps = []

        for analysis_type, analysis_result in completeness_analysis.items():
            for gap in analysis_result.get("gaps", []):
                if gap.get("critical", False):
                    critical_gaps.append(f"{analysis_type}: {gap['message']}")

        return critical_gaps

    def _generate_improvement_suggestions(self, completeness_analysis: Dict[str, Any], query_mode: str) -> List[str]:
        """Generate specific suggestions for improving completeness"""

        suggestions = []

        # Context improvements
        context_analysis = completeness_analysis.get("context_completeness", {})
        if context_analysis.get("missing_required"):
            suggestions.append("Provide more specific vehicle information (manufacturer, model, year)")

        # Source improvements
        source_analysis = completeness_analysis.get("source_completeness", {})
        if source_analysis.get("source_count", 0) < source_analysis.get("minimum_required", 2):
            suggestions.append("Add more authoritative sources")

        if source_analysis.get("missing_preferred_types"):
            missing_types = source_analysis["missing_preferred_types"]
            suggestions.append(f"Include sources from: {', '.join(missing_types)}")

        # Information type improvements
        info_analysis = completeness_analysis.get("information_type_completeness", {})
        if info_analysis.get("missing_info_types"):
            missing_types = info_analysis["missing_info_types"]
            suggestions.append(f"Find sources containing: {', '.join(missing_types)}")

        # Coverage improvements
        coverage_analysis = completeness_analysis.get("coverage_completeness", {})
        if coverage_analysis.get("missing_aspects"):
            missing_aspects = coverage_analysis["missing_aspects"]
            suggestions.append(f"Address these query aspects: {', '.join(missing_aspects)}")

        return suggestions

    def _create_unverifiable_result(self, start_time: datetime, precondition_result: PreconditionResult) -> ValidationStepResult:
        """Create result for unverifiable validation"""

        return ValidationStepResult(
            step_id=f"completeness_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Completeness Analysis",
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
                    message="Cannot analyze completeness",
                    explanation=precondition_result.failure_reason,
                    suggestion="Ensure documents are available for analysis"
                )
            ]
        )

    def _create_error_result(self, start_time: datetime, error_message: str) -> ValidationStepResult:
        """Create result for validation errors"""

        return ValidationStepResult(
            step_id=f"completeness_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Completeness Analysis",
            status=ValidationStatus.FAILED,
            confidence_impact=-5.0,
            summary=f"Analysis failed: {error_message}",
            details={"error": error_message},
            started_at=start_time,
            completed_at=datetime.now(),
            warnings=[
                ValidationWarning(
                    category="validation_error",
                    severity="critical",
                    message="Completeness analysis encountered an error",
                    explanation=error_message,
                    suggestion="Check logs and retry analysis"
                )
            ]
        )