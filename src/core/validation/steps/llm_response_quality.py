"""
LLM Response Quality Validator
Validates quality and accuracy of LLM-generated responses
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import Counter

from ..models.validation_models import (
    ValidationStepResult, ValidationStatus, ValidationStepType,
    ValidationContext, ValidationWarning
)
from .steps_readiness_checker import MetaValidator, PreconditionResult

logger = logging.getLogger(__name__)


class LLMResponseQualityValidator:
    """
    Validates the quality and accuracy of LLM-generated responses
    """

    def __init__(self, step_config: Dict[str, Any], meta_validator: MetaValidator):
        self.step_config = step_config
        self.meta_validator = meta_validator
        self.step_type = ValidationStepType.LLM_INFERENCE

        # Define quality assessment criteria
        self.quality_criteria = self._initialize_quality_criteria()
        self.factual_patterns = self._initialize_factual_patterns()

    def _initialize_quality_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Initialize quality assessment criteria for different query modes"""
        return {
            "facts": {
                "required_elements": ["specific_data", "citations", "factual_claims"],
                "quality_indicators": ["numerical_precision", "technical_accuracy", "source_attribution"],
                "avoid_patterns": ["vague_statements", "unsupported_claims", "generalizations"]
            },
            "features": {
                "required_elements": ["feature_descriptions", "comparisons", "specific_examples"],
                "quality_indicators": ["detailed_explanations", "balanced_coverage", "practical_relevance"],
                "avoid_patterns": ["superficial_descriptions", "missing_comparisons", "biased_language"]
            },
            "tradeoffs": {
                "required_elements": ["advantages", "disadvantages", "balanced_analysis"],
                "quality_indicators": ["objective_tone", "evidence_support", "practical_implications"],
                "avoid_patterns": ["one_sided_analysis", "unsupported_opinions", "emotional_language"]
            },
            "scenarios": {
                "required_elements": ["use_case_analysis", "recommendations", "context_consideration"],
                "quality_indicators": ["practical_advice", "scenario_specificity", "realistic_examples"],
                "avoid_patterns": ["generic_advice", "unrealistic_scenarios", "missing_context"]
            },
            "debate": {
                "required_elements": ["multiple_perspectives", "evidence", "expert_opinions"],
                "quality_indicators": ["balanced_representation", "credible_sources", "logical_structure"],
                "avoid_patterns": ["biased_presentation", "missing_perspectives", "weak_evidence"]
            },
            "quotes": {
                "required_elements": ["user_experiences", "specific_examples", "real_world_context"],
                "quality_indicators": ["authentic_voice", "detailed_experiences", "relatable_situations"],
                "avoid_patterns": ["generic_testimonials", "fabricated_quotes", "unrealistic_experiences"]
            }
        }

    def _initialize_factual_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting factual claims that need verification"""
        return {
            "numerical_claims": [
                r'(\d+)\s*mpg',
                r'(\d+)\s*(?:hp|horsepower)',
                r'(\d+)\s*(?:lb-ft|torque)',
                r'\$(\d+,?\d+)',
                r'(\d+)\s*inches?',
                r'(\d+,?\d+)\s*(?:lbs?|pounds)',
                r'(\d+\.?\d*)\s*seconds?'
            ],
            "rating_claims": [
                r'(\d+)\s*star',
                r'(\d+)/10',
                r'(\d+)%',
                r'rated\s+(\d+)',
                r'score[sd]?\s+(\d+)'
            ],
            "comparative_claims": [
                r'better than',
                r'worse than',
                r'superior to',
                r'inferior to',
                r'outperforms',
                r'exceeds',
                r'falls short'
            ],
            "absolute_claims": [
                r'best',
                r'worst',
                r'highest',
                r'lowest',
                r'most',
                r'least',
                r'always',
                r'never',
                r'all',
                r'none'
            ]
        }

    async def execute(self, context: ValidationContext) -> ValidationStepResult:
        """Execute LLM response quality validation"""

        start_time = datetime.now()

        # Check preconditions
        precondition_result = await self.meta_validator.check_preconditions(
            self.step_type, context, self.step_config
        )

        if precondition_result.status != "READY":
            return self._create_unverifiable_result(start_time, precondition_result)

        try:
            # Perform LLM response quality validation
            result = await self._perform_validation(context)
            result.completed_at = datetime.now()
            result.duration_ms = int((result.completed_at - start_time).total_seconds() * 1000)

            return result

        except Exception as e:
            logger.error(f"LLM response quality validation failed: {str(e)}")
            return self._create_error_result(start_time, str(e))

    async def _perform_validation(self, context: ValidationContext) -> ValidationStepResult:
        """Perform the actual LLM response quality validation"""

        # Get the generated response from context
        generated_response = context.processing_metadata.get("generated_answer", "")

        if not generated_response:
            return self._create_no_response_result(datetime.now())

        documents = context.documents
        query = context.query_text
        query_mode = context.query_mode
        warnings = []

        # Analyze different aspects of response quality
        quality_analysis = {
            "content_quality": self._analyze_content_quality(generated_response, query_mode),
            "factual_accuracy": self._analyze_factual_accuracy(generated_response, documents),
            "source_attribution": self._analyze_source_attribution(generated_response, documents),
            "response_completeness": self._analyze_response_completeness(generated_response, query, query_mode),
            "language_quality": self._analyze_language_quality(generated_response),
            "mode_compliance": self._analyze_mode_compliance(generated_response, query_mode),
            "hallucination_detection": self._detect_potential_hallucinations(generated_response, documents)
        }

        # Calculate component scores
        content_score = quality_analysis["content_quality"]["score"]
        accuracy_score = quality_analysis["factual_accuracy"]["score"]
        attribution_score = quality_analysis["source_attribution"]["score"]
        completeness_score = quality_analysis["response_completeness"]["score"]
        language_score = quality_analysis["language_quality"]["score"]
        compliance_score = quality_analysis["mode_compliance"]["score"]
        hallucination_score = quality_analysis["hallucination_detection"]["score"]

        # Weight the scores based on importance
        weights = {
            "content": 0.2,
            "accuracy": 0.25,
            "attribution": 0.15,
            "completeness": 0.15,
            "language": 0.1,
            "compliance": 0.1,
            "hallucination": 0.05
        }

        overall_quality = (
                content_score * weights["content"] +
                accuracy_score * weights["accuracy"] +
                attribution_score * weights["attribution"] +
                completeness_score * weights["completeness"] +
                language_score * weights["language"] +
                compliance_score * weights["compliance"] +
                hallucination_score * weights["hallucination"]
        )

        # Generate warnings for quality issues
        for analysis_type, analysis_result in quality_analysis.items():
            for issue in analysis_result.get("issues", []):
                severity = "critical" if issue.get("severity") == "high" else "caution"
                warnings.append(ValidationWarning(
                    category=f"llm_{analysis_type}",
                    severity=severity,
                    message=issue["message"],
                    explanation=issue.get("explanation", ""),
                    suggestion=issue.get("suggestion", "Review and improve response quality")
                ))

        # Check for critical quality issues
        critical_issues = [
            issue for analysis in quality_analysis.values()
            for issue in analysis.get("issues", [])
            if issue.get("severity") == "high"
        ]

        # Determine validation status and confidence impact
        if overall_quality >= 0.85 and not critical_issues:
            status = ValidationStatus.PASSED
            confidence_impact = 10.0 + (overall_quality - 0.85) * 20  # 10-13 point boost
        elif overall_quality >= 0.7:
            status = ValidationStatus.WARNING
            confidence_impact = 5.0 + (overall_quality - 0.7) * 25  # 5-8.75 point boost
        else:
            status = ValidationStatus.WARNING
            confidence_impact = max(-8.0, (overall_quality - 0.5) * 15)  # Up to -8 point penalty

            if critical_issues:
                confidence_impact -= len(critical_issues) * 2.0  # Additional penalty for critical issues

            warnings.append(ValidationWarning(
                category="overall_llm_quality",
                severity="critical",
                message="LLM response quality is insufficient",
                explanation=f"Overall quality: {overall_quality:.1%}, Critical issues: {len(critical_issues)}",
                suggestion="Regenerate response with improved prompts and validation"
            ))

        # Build summary
        response_length = len(generated_response)
        factual_claims_count = sum(len(analysis_result.get("claims", [])) for analysis_result in [
            quality_analysis["factual_accuracy"]
        ])

        summary = (f"Analyzed LLM response ({response_length} chars). "
                   f"Overall quality: {overall_quality:.1%}. "
                   f"Accuracy: {accuracy_score:.1%}, "
                   f"Completeness: {completeness_score:.1%}, "
                   f"Critical issues: {len(critical_issues)}")

        # Build detailed results
        details = {
            "response_length": response_length,
            "query_mode": query_mode,
            "overall_quality_score": overall_quality,
            "component_scores": {
                "content_quality": content_score,
                "factual_accuracy": accuracy_score,
                "source_attribution": attribution_score,
                "response_completeness": completeness_score,
                "language_quality": language_score,
                "mode_compliance": compliance_score,
                "hallucination_score": hallucination_score
            },
            "quality_analysis": quality_analysis,
            "critical_issues_count": len(critical_issues),
            "factual_claims_analyzed": factual_claims_count,
            "improvement_recommendations": self._generate_improvement_recommendations(quality_analysis)
        }

        return ValidationStepResult(
            step_id=f"llm_response_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="LLM Response Quality Analysis",
            status=status,
            confidence_impact=confidence_impact,
            summary=summary,
            details=details,
            started_at=datetime.now(),
            warnings=warnings,
            sources_used=[f"response_analysis"]
        )

    def _analyze_content_quality(self, response: str, query_mode: str) -> Dict[str, Any]:
        """Analyze the content quality of the response"""

        criteria = self.quality_criteria.get(query_mode, self.quality_criteria["facts"])
        required_elements = criteria["required_elements"]
        quality_indicators = criteria["quality_indicators"]
        avoid_patterns = criteria["avoid_patterns"]

        issues = []
        score_components = {}

        # Check for required elements
        elements_present = 0
        for element in required_elements:
            if self._check_element_present(response, element):
                elements_present += 1
            else:
                issues.append({
                    "message": f"Missing required element: {element}",
                    "explanation": f"Response should include {element} for {query_mode} mode",
                    "suggestion": f"Add {element} to improve response completeness",
                    "severity": "high"
                })

        score_components["required_elements"] = elements_present / len(required_elements) if required_elements else 1.0

        # Check for quality indicators
        indicators_present = 0
        for indicator in quality_indicators:
            if self._check_quality_indicator(response, indicator):
                indicators_present += 1

        score_components["quality_indicators"] = indicators_present / len(
            quality_indicators) if quality_indicators else 1.0

        # Check for patterns to avoid
        avoid_violations = 0
        for pattern in avoid_patterns:
            if self._check_avoid_pattern(response, pattern):
                avoid_violations += 1
                issues.append({
                    "message": f"Contains pattern to avoid: {pattern}",
                    "explanation": f"Response contains {pattern} which reduces quality",
                    "suggestion": f"Remove or improve {pattern} in response",
                    "severity": "medium"
                })

        score_components["avoid_compliance"] = max(0.0, 1.0 - (
                    avoid_violations / len(avoid_patterns))) if avoid_patterns else 1.0

        # Calculate overall content quality score
        content_score = (
                score_components["required_elements"] * 0.5 +
                score_components["quality_indicators"] * 0.3 +
                score_components["avoid_compliance"] * 0.2
        )

        return {
            "score": content_score,
            "score_components": score_components,
            "elements_present": elements_present,
            "indicators_present": indicators_present,
            "avoid_violations": avoid_violations,
            "issues": issues
        }

    def _check_element_present(self, response: str, element: str) -> bool:
        """Check if a required element is present in the response"""

        element_patterns = {
            "specific_data": [r'\d+', r'specification', r'exactly', r'precisely'],
            "citations": [r'\[', r'\(', r'source', r'according to', r'from'],
            "factual_claims": [r'mpg', r'horsepower', r'price', r'rating', r'score'],
            "feature_descriptions": [r'feature', r'includes?', r'equipped', r'comes with'],
            "comparisons": [r'vs\.?', r'versus', r'compared? to', r'better', r'worse'],
            "specific_examples": [r'example', r'for instance', r'such as', r'like'],
            "advantages": [r'pros?', r'advantages?', r'benefits?', r'positive'],
            "disadvantages": [r'cons?', r'disadvantages?', r'drawbacks?', r'negative'],
            "balanced_analysis": [r'however', r'but', r'on the other hand', r'although'],
            "use_case_analysis": [r'use case', r'scenario', r'situation', r'purpose'],
            "recommendations": [r'recommend', r'suggest', r'should', r'consider'],
            "context_consideration": [r'depend', r'context', r'situation', r'circumstance'],
            "multiple_perspectives": [r'expert', r'analyst', r'opinion', r'perspective'],
            "evidence": [r'study', r'research', r'data', r'test', r'evidence'],
            "expert_opinions": [r'expert', r'professional', r'specialist', r'authority'],
            "user_experiences": [r'owner', r'user', r'customer', r'experience'],
            "real_world_context": [r'real world', r'actual', r'practical', r'everyday']
        }

        patterns = element_patterns.get(element, [element])
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in patterns)

    def _check_quality_indicator(self, response: str, indicator: str) -> bool:
        """Check if a quality indicator is present in the response"""

        indicator_patterns = {
            "numerical_precision": [r'\d+\.\d+', r'\d+,\d+', r'exactly \d+', r'precisely \d+'],
            "technical_accuracy": [r'specification', r'technical', r'engineering', r'measurement'],
            "source_attribution": [r'according to', r'source', r'from', r'\[.*\]', r'\(.*\)'],
            "detailed_explanations": [r'because', r'due to', r'reason', r'explanation', r'detail'],
            "balanced_coverage": [r'both', r'also', r'additionally', r'furthermore', r'however'],
            "practical_relevance": [r'practical', r'useful', r'relevant', r'applicable', r'real'],
            "objective_tone": [r'data shows', r'research indicates', r'studies suggest', r'evidence'],
            "evidence_support": [r'study', r'research', r'test', r'data', r'statistics'],
            "practical_implications": [r'means', r'implies', r'results in', r'consequence'],
            "practical_advice": [r'should', r'recommend', r'suggest', r'advice', r'tip'],
            "scenario_specificity": [r'specific', r'particular', r'exact', r'precise'],
            "realistic_examples": [r'example', r'instance', r'case', r'situation'],
            "balanced_representation": [r'both sides', r'different views', r'various', r'multiple'],
            "credible_sources": [r'expert', r'authority', r'professional', r'specialist'],
            "logical_structure": [r'first', r'second', r'next', r'finally', r'therefore'],
            "authentic_voice": [r'personal', r'experience', r'own', r'myself', r'individual'],
            "relatable_situations": [r'common', r'typical', r'usual', r'normal', r'everyday']
        }

        patterns = indicator_patterns.get(indicator, [indicator])
        return any(re.search(pattern, response, re.IGNORECASE) for pattern in patterns)

    def _check_avoid_pattern(self, response: str, pattern: str) -> bool:
        """Check if a pattern to avoid is present in the response"""

        avoid_patterns = {
            "vague_statements": [r'might be', r'could be', r'possibly', r'perhaps', r'maybe'],
            "unsupported_claims": [r'obviously', r'clearly', r'everyone knows', r'it is known'],
            "generalizations": [r'all cars', r'every vehicle', r'always', r'never', r'most people'],
            "superficial_descriptions": [r'good', r'bad', r'nice', r'great', r'terrible'],
            "missing_comparisons": [r'best car', r'worst option', r'perfect choice'],
            "biased_language": [r'unfortunately', r'luckily', r'sadly', r'happily'],
            "one_sided_analysis": [r'only advantage', r'no disadvantages', r'perfect'],
            "unsupported_opinions": [r'I think', r'I believe', r'in my opinion', r'personally'],
            "emotional_language": [r'amazing', r'terrible', r'wonderful', r'awful', r'love', r'hate'],
            "generic_advice": [r'depends on you', r'personal preference', r'up to you'],
            "unrealistic_scenarios": [r'unlimited budget', r'perfect conditions', r'ideal situation'],
            "missing_context": [r'regardless of', r'in all cases', r'universally'],
            "biased_presentation": [r'supporters claim', r'critics argue', r'opponents say'],
            "missing_perspectives": [r'only view', r'single perspective', r'one side'],
            "weak_evidence": [r'rumor', r'gossip', r'unconfirmed', r'allegedly'],
            "generic_testimonials": [r'customer said', r'user reported', r'owner mentioned'],
            "fabricated_quotes": [r'someone once said', r'anonymous user', r'unnamed source'],
            "unrealistic_experiences": [r'perfect experience', r'no problems ever', r'always satisfied']
        }

        patterns = avoid_patterns.get(pattern, [pattern])
        return any(re.search(pat, response, re.IGNORECASE) for pat in patterns)

    def _analyze_factual_accuracy(self, response: str, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze factual accuracy of claims in the response"""

        # Extract factual claims from response
        factual_claims = self._extract_factual_claims(response)

        # Verify claims against source documents
        verified_claims = []
        unverified_claims = []
        contradicted_claims = []

        for claim in factual_claims:
            verification_result = self._verify_claim_against_documents(claim, documents)

            if verification_result["status"] == "verified":
                verified_claims.append(claim)
            elif verification_result["status"] == "contradicted":
                contradicted_claims.append(claim)
            else:
                unverified_claims.append(claim)

        # Calculate accuracy score
        total_claims = len(factual_claims)
        if total_claims == 0:
            accuracy_score = 1.0  # No claims to verify
        else:
            verified_count = len(verified_claims)
            contradicted_count = len(contradicted_claims)
            accuracy_score = (verified_count - contradicted_count * 2) / total_claims
            accuracy_score = max(0.0, min(1.0, accuracy_score))

        issues = []

        # Generate issues for contradicted claims
        for claim in contradicted_claims:
            issues.append({
                "message": f"Contradicted factual claim: {claim['text']}",
                "explanation": "Claim contradicts information in source documents",
                "suggestion": "Verify and correct factual information",
                "severity": "high"
            })

        # Generate issues for many unverified claims
        if len(unverified_claims) > len(verified_claims):
            issues.append({
                "message": "Many unverified factual claims",
                "explanation": f"{len(unverified_claims)} claims could not be verified against sources",
                "suggestion": "Ensure claims are supported by source documents",
                "severity": "medium"
            })

        return {
            "score": accuracy_score,
            "total_claims": total_claims,
            "verified_claims": len(verified_claims),
            "unverified_claims": len(unverified_claims),
            "contradicted_claims": len(contradicted_claims),
            "claims": factual_claims,
            "issues": issues
        }

    def _extract_factual_claims(self, response: str) -> List[Dict[str, Any]]:
        """Extract factual claims from the response"""

        claims = []

        # Extract numerical claims
        for pattern in self.factual_patterns["numerical_claims"]:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                claims.append({
                    "type": "numerical",
                    "text": match.group(0),
                    "value": match.group(1) if match.groups() else match.group(0),
                    "context": response[max(0, match.start() - 50):match.end() + 50]
                })

        # Extract rating claims
        for pattern in self.factual_patterns["rating_claims"]:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                claims.append({
                    "type": "rating",
                    "text": match.group(0),
                    "value": match.group(1) if match.groups() else match.group(0),
                    "context": response[max(0, match.start() - 50):match.end() + 50]
                })

        # Extract comparative claims
        for pattern in self.factual_patterns["comparative_claims"]:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                claims.append({
                    "type": "comparative",
                    "text": match.group(0),
                    "context": response[max(0, match.start() - 50):match.end() + 50]
                })

        # Extract absolute claims
        for pattern in self.factual_patterns["absolute_claims"]:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                claims.append({
                    "type": "absolute",
                    "text": match.group(0),
                    "context": response[max(0, match.start() - 50):match.end() + 50]
                })

        return claims

    def _verify_claim_against_documents(self, claim: Dict[str, Any], documents: List[Dict]) -> Dict[str, Any]:
        """Verify a factual claim against source documents"""

        claim_text = claim["text"].lower()
        claim_type = claim["type"]

        supporting_docs = []
        contradicting_docs = []

        for i, doc in enumerate(documents):
            content = doc.get("content", "").lower()

            if claim_type == "numerical":
                # Check for exact numerical matches or close values
                if claim["value"] in content:
                    supporting_docs.append(i)
                elif self._check_numerical_contradiction(claim, content):
                    contradicting_docs.append(i)

            elif claim_type == "comparative":
                # Check for comparative statements
                if any(comp in content for comp in [claim_text, "better than", "worse than"]):
                    supporting_docs.append(i)

            elif claim_type == "absolute":
                # Check for absolute claims (more strict verification)
                if claim_text in content:
                    supporting_docs.append(i)
                elif self._check_absolute_contradiction(claim, content):
                    contradicting_docs.append(i)

            else:
                # General text matching
                if claim_text in content:
                    supporting_docs.append(i)

        # Determine verification status
        if contradicting_docs:
            status = "contradicted"
        elif supporting_docs:
            status = "verified"
        else:
            status = "unverified"

        return {
            "status": status,
            "supporting_documents": supporting_docs,
            "contradicting_documents": contradicting_docs
        }

    def _check_numerical_contradiction(self, claim: Dict[str, Any], content: str) -> bool:
        """Check if numerical claim contradicts information in content"""

        try:
            claim_value = float(claim["value"].replace(",", ""))

            # Look for similar numerical patterns in content
            claim_context = claim.get("context", "").lower()

            if "mpg" in claim_context:
                mpg_matches = re.findall(r'(\d+)\s*mpg', content)
                for match in mpg_matches:
                    doc_value = float(match)
                    if abs(claim_value - doc_value) > 3:  # More than 3 MPG difference
                        return True

            elif "horsepower" in claim_context or "hp" in claim_context:
                hp_matches = re.findall(r'(\d+)\s*(?:hp|horsepower)', content)
                for match in hp_matches:
                    doc_value = float(match)
                    if abs(claim_value - doc_value) > 20:  # More than 20 HP difference
                        return True

        except (ValueError, TypeError):
            pass

        return False

    def _check_absolute_contradiction(self, claim: Dict[str, Any], content: str) -> bool:
        """Check if absolute claim contradicts information in content"""

        claim_text = claim["text"].lower()

        # Check for contradictory absolute statements
        contradictions = {
            "best": ["worst", "inferior", "poor"],
            "worst": ["best", "superior", "excellent"],
            "highest": ["lowest", "minimum"],
            "lowest": ["highest", "maximum"],
            "always": ["never", "sometimes", "rarely"],
            "never": ["always", "often", "frequently"],
            "all": ["none", "some", "few"],
            "none": ["all", "many", "most"]
        }

        if claim_text in contradictions:
            return any(contra in content for contra in contradictions[claim_text])

        return False

    def _analyze_source_attribution(self, response: str, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze source attribution in the response"""

        # Look for citation patterns
        citation_patterns = [
            r'\[.*?\]',  # [1], [source], etc.
            r'\(.*?\)',  # (source), (2023), etc.
            r'according to',
            r'source:',
            r'from',
            r'as reported by'
        ]

        citations_found = []
        for pattern in citation_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            citations_found.extend([match.group(0) for match in matches])

        # Count factual claims that should have citations
        factual_claims = self._extract_factual_claims(response)
        claims_needing_citations = [claim for claim in factual_claims if claim["type"] in ["numerical", "rating"]]

        # Calculate attribution score
        if not claims_needing_citations:
            attribution_score = 1.0  # No claims needing citations
        else:
            # Rough estimate: assume each citation covers some claims
            citation_coverage = min(1.0, len(citations_found) / len(claims_needing_citations))
            attribution_score = citation_coverage

        issues = []

        if len(claims_needing_citations) > len(citations_found):
            issues.append({
                "message": "Insufficient source attribution",
                "explanation": f"{len(claims_needing_citations)} claims need citations, only {len(citations_found)} found",
                "suggestion": "Add proper citations for factual claims",
                "severity": "medium"
            })

        return {
            "score": attribution_score,
            "citations_found": len(citations_found),
            "claims_needing_citations": len(claims_needing_citations),
            "citation_patterns": citations_found,
            "issues": issues
        }

    def _analyze_response_completeness(self, response: str, query: str, query_mode: str) -> Dict[str, Any]:
        """Analyze completeness of the response relative to the query"""

        # Extract key aspects from query
        query_aspects = self._extract_query_aspects(query, query_mode)

        # Check which aspects are addressed in response
        addressed_aspects = []
        for aspect in query_aspects:
            if self._aspect_addressed_in_response(aspect, response):
                addressed_aspects.append(aspect)

        # Calculate completeness score
        completeness_score = len(addressed_aspects) / len(query_aspects) if query_aspects else 1.0

        missing_aspects = [aspect for aspect in query_aspects if aspect not in addressed_aspects]

        issues = []

        if completeness_score < 0.7:
            issues.append({
                "message": "Incomplete response to query",
                "explanation": f"Only {len(addressed_aspects)}/{len(query_aspects)} query aspects addressed",
                "suggestion": f"Address missing aspects: {', '.join(missing_aspects[:3])}",
                "severity": "high"
            })

        return {
            "score": completeness_score,
            "query_aspects": query_aspects,
            "addressed_aspects": addressed_aspects,
            "missing_aspects": missing_aspects,
            "issues": issues
        }

    def _extract_query_aspects(self, query: str, query_mode: str) -> List[str]:
        """Extract key aspects from the query that should be addressed"""

        aspects = []
        query_lower = query.lower()

        # Extract explicit aspects from query
        if "mpg" in query_lower or "fuel" in query_lower:
            aspects.append("fuel_economy")
        if "horsepower" in query_lower or "power" in query_lower:
            aspects.append("performance")
        if "price" in query_lower or "cost" in query_lower:
            aspects.append("pricing")
        if "safety" in query_lower or "crash" in query_lower:
            aspects.append("safety")
        if "features" in query_lower or "equipment" in query_lower:
            aspects.append("features")
        if "reliability" in query_lower or "dependable" in query_lower:
            aspects.append("reliability")

        # Add mode-specific aspects
        mode_aspects = {
            "facts": ["specifications", "technical_details"],
            "features": ["feature_comparison", "equipment_details"],
            "tradeoffs": ["advantages", "disadvantages"],
            "scenarios": ["use_cases", "recommendations"],
            "debate": ["different_opinions", "evidence"],
            "quotes": ["user_experiences", "real_world_feedback"]
        }

        aspects.extend(mode_aspects.get(query_mode, []))

        # Add general aspects if none specific found
        if not aspects:
            aspects = ["general_information"]

        return list(set(aspects))

    def _aspect_addressed_in_response(self, aspect: str, response: str) -> bool:
        """Check if an aspect is addressed in the response"""

        aspect_keywords = {
            "fuel_economy": ["mpg", "fuel", "economy", "efficiency", "gas"],
            "performance": ["horsepower", "torque", "acceleration", "power", "performance"],
            "pricing": ["price", "cost", "expensive", "affordable", "value", "$"],
            "safety": ["safety", "crash", "airbag", "protection", "secure"],
            "features": ["features", "equipment", "includes", "comes with"],
            "reliability": ["reliability", "dependable", "durable", "problems"],
            "specifications": ["specifications", "specs", "technical", "details"],
            "technical_details": ["engine", "transmission", "dimensions", "weight"],
            "feature_comparison": ["compared", "versus", "better", "different"],
            "equipment_details": ["standard", "optional", "available", "equipped"],
            "advantages": ["advantages", "pros", "benefits", "positive"],
            "disadvantages": ["disadvantages", "cons", "drawbacks", "negative"],
            "use_cases": ["use", "purpose", "scenario", "ideal for"],
            "recommendations": ["recommend", "suggest", "should", "consider"],
            "different_opinions": ["opinion", "perspective", "view", "expert"],
            "evidence": ["evidence", "data", "study", "research"],
            "user_experiences": ["experience", "owner", "user", "customer"],
            "real_world_feedback": ["real world", "actual", "practical"],
            "general_information": ["information", "about", "details"]
        }

        keywords = aspect_keywords.get(aspect, [aspect])
        return any(keyword in response.lower() for keyword in keywords)

    def _analyze_language_quality(self, response: str) -> Dict[str, Any]:
        """Analyze language quality of the response"""

        issues = []
        score_components = {}

        # Check response length
        response_length = len(response)
        if response_length < 100:
            issues.append({
                "message": "Response too short",
                "explanation": f"Response is only {response_length} characters",
                "suggestion": "Provide more detailed and comprehensive response",
                "severity": "medium"
            })
            score_components["length"] = 0.5
        elif response_length > 2000:
            issues.append({
                "message": "Response may be too long",
                "explanation": f"Response is {response_length} characters",
                "suggestion": "Consider making response more concise",
                "severity": "low"
            })
            score_components["length"] = 0.8
        else:
            score_components["length"] = 1.0

        # Check for structure indicators
        structure_indicators = [
            "1.", "2.", "3.",  # Numbered lists
            "-", "•",  # Bullet points
            "**", "__",  # Bold formatting
            ":",  # Colons for definitions
        ]

        structure_count = sum(1 for indicator in structure_indicators if indicator in response)
        score_components["structure"] = min(1.0, structure_count / 3)  # Expect at least 3 structure elements

        # Check for readability indicators
        sentences = response.split('.')
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0

        if avg_sentence_length > 25:
            issues.append({
                "message": "Sentences may be too long",
                "explanation": f"Average sentence length: {avg_sentence_length:.1f} words",
                "suggestion": "Use shorter, clearer sentences",
                "severity": "low"
            })
            score_components["readability"] = 0.7
        else:
            score_components["readability"] = 1.0

        # Check for clarity indicators
        clarity_indicators = ["because", "therefore", "however", "for example", "such as"]
        clarity_count = sum(1 for indicator in clarity_indicators if indicator.lower() in response.lower())
        score_components["clarity"] = min(1.0, clarity_count / 2)  # Expect at least 2 clarity indicators

        # Calculate overall language quality
        language_score = (
                score_components["length"] * 0.3 +
                score_components["structure"] * 0.3 +
                score_components["readability"] * 0.2 +
                score_components["clarity"] * 0.2
        )

        return {
            "score": language_score,
            "score_components": score_components,
            "response_length": response_length,
            "average_sentence_length": avg_sentence_length,
            "structure_indicators": structure_count,
            "clarity_indicators": clarity_count,
            "issues": issues
        }

    def _analyze_mode_compliance(self, response: str, query_mode: str) -> Dict[str, Any]:
        """Analyze compliance with query mode requirements"""

        mode_requirements = {
            "facts": {
                "should_contain": ["specific", "data", "number", "specification"],
                "should_avoid": ["opinion", "I think", "might be", "possibly"]
            },
            "features": {
                "should_contain": ["feature", "includes", "equipped", "available"],
                "should_avoid": ["best", "worst", "always", "never"]
            },
            "tradeoffs": {
                "should_contain": ["advantage", "disadvantage", "pros", "cons", "however"],
                "should_avoid": ["perfect", "no drawbacks", "only benefits"]
            },
            "scenarios": {
                "should_contain": ["scenario", "use case", "situation", "recommend"],
                "should_avoid": ["always ideal", "perfect for everyone"]
            },
            "debate": {
                "should_contain": ["perspective", "opinion", "expert", "argue"],
                "should_avoid": ["definitely", "obviously", "clearly best"]
            },
            "quotes": {
                "should_contain": ["experience", "owner", "user", "feedback"],
                "should_avoid": ["all users", "everyone says", "universally"]
            }
        }

        requirements = mode_requirements.get(query_mode, {"should_contain": [], "should_avoid": []})

        issues = []
        score_components = {}

        # Check for required elements
        should_contain = requirements["should_contain"]
        contained_count = sum(1 for element in should_contain if element.lower() in response.lower())
        score_components["required_elements"] = contained_count / len(should_contain) if should_contain else 1.0

        if score_components["required_elements"] < 0.5:
            issues.append({
                "message": f"Response doesn't match {query_mode} mode requirements",
                "explanation": f"Only {contained_count}/{len(should_contain)} required elements found",
                "suggestion": f"Include elements like: {', '.join(should_contain[:3])}",
                "severity": "medium"
            })

        # Check for elements to avoid
        should_avoid = requirements["should_avoid"]
        avoided_count = sum(1 for element in should_avoid if element.lower() in response.lower())
        score_components["avoided_elements"] = max(0.0,
                                                   1.0 - (avoided_count / len(should_avoid))) if should_avoid else 1.0

        if avoided_count > 0:
            issues.append({
                "message": f"Response contains elements to avoid for {query_mode} mode",
                "explanation": f"{avoided_count} problematic elements found",
                "suggestion": f"Avoid elements like: {', '.join(should_avoid[:3])}",
                "severity": "low"
            })

        # Calculate overall compliance score
        compliance_score = (score_components["required_elements"] * 0.7 + score_components["avoided_elements"] * 0.3)

        return {
            "score": compliance_score,
            "score_components": score_components,
            "required_elements_found": contained_count,
            "problematic_elements_found": avoided_count,
            "issues": issues
        }

    def _detect_potential_hallucinations(self, response: str, documents: List[Dict]) -> Dict[str, Any]:
        """Detect potential hallucinations in the response"""

        issues = []
        hallucination_indicators = []

        # Check for overly specific claims without support
        specific_patterns = [
            r'\d+\.\d+\d+',  # Very precise decimals
            r'exactly \d+',  # Claims of exactness
            r'precisely \d+',  # Claims of precision
            r'studies show that exactly',  # Overly specific study claims
            r'research proves that',  # Overly strong research claims
        ]

        for pattern in specific_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                hallucination_indicators.append({
                    "type": "overly_specific",
                    "text": match.group(0),
                    "context": response[max(0, match.start() - 30):match.end() + 30]
                })

        # Check for claims that seem fabricated
        fabrication_patterns = [
            r'a study from \d{4} found',
            r'research by [A-Z][a-z]+ University',
            r'according to a recent survey',
            r'data from \w+ Research shows',
        ]

        for pattern in fabrication_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                # Check if this study/research is mentioned in documents
                study_mentioned = any(match.group(0).lower() in doc.get("content", "").lower()
                                      for doc in documents)
                if not study_mentioned:
                    hallucination_indicators.append({
                        "type": "unsupported_study",
                        "text": match.group(0),
                        "context": response[max(0, match.start() - 30):match.end() + 30]
                    })

        # Check for impossible combinations
        impossibility_checks = self._check_impossibilities(response)
        hallucination_indicators.extend(impossibility_checks)

        # Generate issues
        if len(hallucination_indicators) > 0:
            issues.append({
                "message": "Potential hallucinations detected",
                "explanation": f"{len(hallucination_indicators)} suspicious claims found",
                "suggestion": "Verify all claims against source documents",
                "severity": "high"
            })

        # Calculate hallucination score (higher is better)
        if len(hallucination_indicators) == 0:
            hallucination_score = 1.0
        elif len(hallucination_indicators) <= 2:
            hallucination_score = 0.7
        else:
            hallucination_score = max(0.0, 0.5 - (len(hallucination_indicators) - 2) * 0.1)

        return {
            "score": hallucination_score,
            "hallucination_indicators": hallucination_indicators,
            "potential_hallucinations": len(hallucination_indicators),
            "issues": issues
        }

    def _check_impossibilities(self, response: str) -> List[Dict[str, Any]]:
        """Check for physically impossible claims"""

        impossibilities = []

        # Check for impossible fuel economy claims
        mpg_matches = re.finditer(r'(\d+)\s*mpg', response, re.IGNORECASE)
        for match in mpg_matches:
            mpg_value = int(match.group(1))
            if mpg_value > 200:  # Impossible for regular cars
                impossibilities.append({
                    "type": "impossible_mpg",
                    "text": match.group(0),
                    "context": response[max(0, match.start() - 30):match.end() + 30]
                })

        # Check for impossible horsepower claims
        hp_matches = re.finditer(r'(\d+)\s*(?:hp|horsepower)', response, re.IGNORECASE)
        for match in hp_matches:
            hp_value = int(match.group(1))
            if hp_value > 2000:  # Extreme for road cars
                impossibilities.append({
                    "type": "extreme_horsepower",
                    "text": match.group(0),
                    "context": response[max(0, match.start() - 30):match.end() + 30]
                })

        # Check for impossible price claims
        price_matches = re.finditer(r'\$(\d+,?\d+)', response)
        for match in price_matches:
            price_str = match.group(1).replace(',', '')
            try:
                price_value = int(price_str)
                if price_value > 1000000:  # Very expensive for most cars
                    impossibilities.append({
                        "type": "extreme_price",
                        "text": match.group(0),
                        "context": response[max(0, match.start() - 30):match.end() + 30]
                    })
            except ValueError:
                pass

        return impossibilities

    def _generate_improvement_recommendations(self, quality_analysis: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for improving response quality"""

        recommendations = []

        # Content quality recommendations
        content_analysis = quality_analysis["content_quality"]
        if content_analysis["score"] < 0.8:
            if content_analysis["elements_present"] < len(content_analysis.get("required_elements", [])):
                recommendations.append("Include all required elements for the query mode")
            if content_analysis["avoid_violations"] > 0:
                recommendations.append("Remove vague or unsupported statements")

        # Factual accuracy recommendations
        accuracy_analysis = quality_analysis["factual_accuracy"]
        if accuracy_analysis["score"] < 0.8:
            if accuracy_analysis["contradicted_claims"] > 0:
                recommendations.append("Correct factual claims that contradict source documents")
            if accuracy_analysis["unverified_claims"] > accuracy_analysis["verified_claims"]:
                recommendations.append("Ensure all factual claims are supported by source documents")

        # Source attribution recommendations
        attribution_analysis = quality_analysis["source_attribution"]
        if attribution_analysis["score"] < 0.7:
            recommendations.append("Add proper citations for factual claims and statistics")

        # Completeness recommendations
        completeness_analysis = quality_analysis["response_completeness"]
        if completeness_analysis["score"] < 0.8:
            missing = completeness_analysis.get("missing_aspects", [])
            if missing:
                recommendations.append(f"Address missing query aspects: {', '.join(missing[:3])}")

        # Language quality recommendations
        language_analysis = quality_analysis["language_quality"]
        if language_analysis["score"] < 0.8:
            if language_analysis["response_length"] < 200:
                recommendations.append("Provide more detailed and comprehensive response")
            if language_analysis.get("structure_indicators", 0) < 2:
                recommendations.append("Improve response structure with headings and bullet points")

        # Hallucination recommendations
        hallucination_analysis = quality_analysis["hallucination_detection"]
        if hallucination_analysis["score"] < 0.9:
            recommendations.append("Verify all specific claims and remove unsupported assertions")

        return recommendations

    def _create_no_response_result(self, start_time: datetime) -> ValidationStepResult:
        """Create result when no response is provided"""

        return ValidationStepResult(
            step_id=f"llm_response_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="LLM Response Quality Analysis",
            status=ValidationStatus.FAILED,
            confidence_impact=-15.0,
            summary="No LLM response found for analysis",
            details={"error": "No generated response provided"},
            started_at=start_time,
            completed_at=datetime.now(),
            warnings=[
                ValidationWarning(
                    category="no_response",
                    severity="critical",
                    message="No LLM response available for validation",
                    explanation="Cannot validate response quality without a generated response",
                    suggestion="Ensure LLM generates a response before validation"
                )
            ]
        )

    def _create_unverifiable_result(self, start_time: datetime,
                                    precondition_result: PreconditionResult) -> ValidationStepResult:
        """Create result for unverifiable validation"""

        return ValidationStepResult(
            step_id=f"llm_response_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="LLM Response Quality Analysis",
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
                    message="Cannot validate LLM response quality",
                    explanation=precondition_result.failure_reason,
                    suggestion="Ensure LLM response and source documents are available"
                )
            ]
        )

    def _create_error_result(self, start_time: datetime, error_message: str) -> ValidationStepResult:
        """Create result for validation errors"""

        return ValidationStepResult(
            step_id=f"llm_response_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="LLM Response Quality Analysis",
            status=ValidationStatus.FAILED,
            confidence_impact=-10.0,
            summary=f"Analysis failed: {error_message}",
            details={"error": error_message},
            started_at=start_time,
            completed_at=datetime.now(),
            warnings=[
                ValidationWarning(
                    category="validation_error",
                    severity="critical",
                    message="LLM response quality analysis encountered an error",
                    explanation=error_message,
                    suggestion="Check logs and retry analysis"
                )
            ]
        )