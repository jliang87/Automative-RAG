"""
Retrieval Quality Validator
Validates retrieval quality and relevance for automotive queries
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from collections import Counter

from src.models import (
    ValidationStepResult, ValidationStatus, ValidationStepType,
    ValidationContext, ValidationWarning
)
from .steps_readiness_checker import MetaValidator, PreconditionResult

logger = logging.getLogger(__name__)


class RetrievalQualityValidator:
    """
    Validates the quality and relevance of document retrieval
    """

    def __init__(self, step_config: Dict[str, Any], meta_validator: MetaValidator):
        self.step_config = step_config
        self.meta_validator = meta_validator
        self.step_type = ValidationStepType.RETRIEVAL

        # Define automotive-specific keywords and relevance patterns
        self.automotive_keywords = self._initialize_automotive_keywords()
        self.relevance_patterns = self._initialize_relevance_patterns()

    def _initialize_automotive_keywords(self) -> Dict[str, List[str]]:
        """Initialize automotive-specific keywords for relevance assessment"""
        return {
            "vehicles": [
                "car", "truck", "suv", "sedan", "coupe", "hatchback", "wagon",
                "crossover", "minivan", "pickup", "convertible", "hybrid", "electric"
            ],
            "manufacturers": [
                "toyota", "honda", "ford", "chevrolet", "nissan", "hyundai", "kia",
                "bmw", "mercedes", "audi", "volkswagen", "subaru", "mazda", "lexus",
                "acura", "infiniti", "cadillac", "lincoln", "buick", "gmc", "jeep",
                "ram", "chrysler", "dodge", "mitsubishi", "volvo", "jaguar", "land rover",
                "tesla", "porsche", "ferrari", "lamborghini", "maserati"
            ],
            "specifications": [
                "mpg", "horsepower", "torque", "engine", "transmission", "fuel economy",
                "acceleration", "top speed", "weight", "dimensions", "cargo", "seating",
                "towing capacity", "ground clearance", "wheelbase"
            ],
            "features": [
                "infotainment", "navigation", "bluetooth", "usb", "backup camera",
                "blind spot", "lane assist", "cruise control", "air conditioning",
                "heated seats", "sunroof", "all wheel drive", "four wheel drive"
            ],
            "safety": [
                "airbag", "abs", "stability control", "traction control", "nhtsa",
                "iihs", "crash test", "safety rating", "collision avoidance"
            ],
            "maintenance": [
                "warranty", "service", "maintenance", "oil change", "tire rotation",
                "brake", "repair", "dealership", "parts", "reliability"
            ]
        }

    def _initialize_relevance_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for assessing query-document relevance"""
        return {
            "facts": [
                r'specification', r'mpg', r'horsepower', r'torque', r'price',
                r'dimensions', r'weight', r'engine', r'transmission'
            ],
            "features": [
                r'features?', r'equipment', r'options?', r'technology',
                r'infotainment', r'safety features?', r'comfort'
            ],
            "tradeoffs": [
                r'pros?', r'cons?', r'advantages?', r'disadvantages?',
                r'trade-?offs?', r'compared? to', r'vs\.?', r'versus'
            ],
            "scenarios": [
                r'use case', r'scenario', r'situation', r'best for',
                r'ideal for', r'suitable for', r'recommended for'
            ],
            "debate": [
                r'opinion', r'debate', r'discussion', r'expert',
                r'analyst', r'review', r'perspective', r'viewpoint'
            ],
            "quotes": [
                r'owner', r'experience', r'review', r'feedback',
                r'user', r'customer', r'real world', r'long term'
            ]
        }

    async def execute(self, context: ValidationContext) -> ValidationStepResult:
        """Execute retrieval quality validation"""

        start_time = datetime.now()

        # Check preconditions
        precondition_result = await self.meta_validator.check_preconditions(
            self.step_type, context, self.step_config
        )

        if precondition_result.status != "READY":
            return self._create_unverifiable_result(start_time, precondition_result)

        try:
            # Perform retrieval quality validation
            result = await self._perform_validation(context)
            result.completed_at = datetime.now()
            result.duration_ms = int((result.completed_at - start_time).total_seconds() * 1000)

            return result

        except Exception as e:
            logger.error(f"Retrieval quality validation failed: {str(e)}")
            return self._create_error_result(start_time, str(e))

    async def _perform_validation(self, context: ValidationContext) -> ValidationStepResult:
        """Perform the actual retrieval quality validation"""

        documents = context.documents
        query = context.query_text
        query_mode = context.query_mode
        warnings = []
        sources_analyzed = []

        # Analyze different aspects of retrieval quality
        quality_analysis = {
            "relevance_analysis": self._analyze_document_relevance(documents, query, query_mode),
            "coverage_analysis": self._analyze_query_coverage(documents, query, query_mode),
            "diversity_analysis": self._analyze_retrieval_diversity(documents),
            "automotive_focus_analysis": self._analyze_automotive_focus(documents),
            "quality_distribution": self._analyze_quality_distribution(documents)
        }

        # Calculate retrieval quality scores
        relevance_score = quality_analysis["relevance_analysis"]["average_relevance"]
        coverage_score = quality_analysis["coverage_analysis"]["coverage_score"]
        diversity_score = quality_analysis["diversity_analysis"]["diversity_score"]
        automotive_score = quality_analysis["automotive_focus_analysis"]["automotive_focus_score"]
        quality_score = quality_analysis["quality_distribution"]["average_quality"]

        # Weight the scores
        weights = {
            "relevance": 0.35,
            "coverage": 0.25,
            "diversity": 0.15,
            "automotive": 0.15,
            "quality": 0.10
        }

        overall_retrieval_quality = (
                relevance_score * weights["relevance"] +
                coverage_score * weights["coverage"] +
                diversity_score * weights["diversity"] +
                automotive_score * weights["automotive"] +
                quality_score * weights["quality"]
        )

        # Generate warnings for quality issues
        if relevance_score < 0.7:
            warnings.append(ValidationWarning(
                category="low_relevance",
                severity="critical",
                message="Low document relevance to query",
                explanation=f"Average relevance: {relevance_score:.1%}",
                suggestion="Improve search query or expand document retrieval scope"
            ))

        if coverage_score < 0.6:
            warnings.append(ValidationWarning(
                category="poor_coverage",
                severity="caution",
                message="Query aspects poorly covered by retrieved documents",
                explanation=f"Coverage score: {coverage_score:.1%}",
                suggestion="Add documents addressing missing query aspects"
            ))

        if diversity_score < 0.5:
            warnings.append(ValidationWarning(
                category="low_diversity",
                severity="caution",
                message="Limited diversity in retrieved documents",
                explanation=f"Diversity score: {diversity_score:.1%}",
                suggestion="Include sources from different types and domains"
            ))

        if automotive_score < 0.8:
            warnings.append(ValidationWarning(
                category="automotive_focus",
                severity="caution",
                message="Some documents may not be automotive-focused",
                explanation=f"Automotive focus: {automotive_score:.1%}",
                suggestion="Filter for automotive-specific content"
            ))

        # Determine validation status and confidence impact
        if overall_retrieval_quality >= 0.85:
            status = ValidationStatus.PASSED
            confidence_impact = 8.0 + (overall_retrieval_quality - 0.85) * 20  # 8-11 point boost
        elif overall_retrieval_quality >= 0.7:
            status = ValidationStatus.WARNING
            confidence_impact = 3.0 + (overall_retrieval_quality - 0.7) * 25  # 3-6.75 point boost
        else:
            status = ValidationStatus.WARNING
            confidence_impact = max(-5.0, (overall_retrieval_quality - 0.5) * 15)  # Up to -5 point penalty

            warnings.append(ValidationWarning(
                category="overall_retrieval_quality",
                severity="critical",
                message="Poor overall retrieval quality",
                explanation=f"Quality score: {overall_retrieval_quality:.1%}",
                suggestion="Improve document retrieval strategy and query matching"
            ))

        # Track sources analyzed
        sources_analyzed = [f"document_{i}" for i in range(len(documents))]

        # Build summary
        summary = (f"Analyzed {len(documents)} retrieved documents. "
                   f"Overall quality: {overall_retrieval_quality:.1%}. "
                   f"Relevance: {relevance_score:.1%}, "
                   f"Coverage: {coverage_score:.1%}, "
                   f"Diversity: {diversity_score:.1%}")

        # Build detailed results
        details = {
            "documents_analyzed": len(documents),
            "query_mode": query_mode,
            "overall_retrieval_quality": overall_retrieval_quality,
            "component_scores": {
                "relevance_score": relevance_score,
                "coverage_score": coverage_score,
                "diversity_score": diversity_score,
                "automotive_focus_score": automotive_score,
                "quality_score": quality_score
            },
            "quality_analysis": quality_analysis,
            "retrieval_effectiveness": self._assess_retrieval_effectiveness(quality_analysis),
            "improvement_recommendations": self._generate_retrieval_improvements(quality_analysis, query_mode)
        }

        return ValidationStepResult(
            step_id=f"retrieval_quality_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Retrieval Quality Analysis",
            status=status,
            confidence_impact=confidence_impact,
            summary=summary,
            details=details,
            started_at=start_time,
            warnings=warnings,
            sources_used=sources_analyzed
        )

    def _analyze_document_relevance(self, documents: List[Dict], query: str, query_mode: str) -> Dict[str, Any]:
        """Analyze relevance of retrieved documents to the query"""

        relevance_scores = []
        document_analysis = []

        query_keywords = self._extract_query_keywords(query)
        mode_patterns = self.relevance_patterns.get(query_mode, [])

        for i, doc in enumerate(documents):
            doc_relevance = self._calculate_document_relevance(
                doc, query_keywords, mode_patterns, query_mode
            )
            relevance_scores.append(doc_relevance["score"])
            document_analysis.append({
                "document_index": i,
                "relevance_score": doc_relevance["score"],
                "matched_keywords": doc_relevance["matched_keywords"],
                "mode_pattern_matches": doc_relevance["mode_pattern_matches"],
                "content_length": len(doc.get("content", "")),
                "relevance_factors": doc_relevance["factors"]
            })

        # Calculate statistics
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        high_relevance_docs = sum(1 for score in relevance_scores if score >= 0.8)
        low_relevance_docs = sum(1 for score in relevance_scores if score < 0.5)

        return {
            "average_relevance": avg_relevance,
            "relevance_distribution": {
                "high": high_relevance_docs,
                "medium": len(relevance_scores) - high_relevance_docs - low_relevance_docs,
                "low": low_relevance_docs
            },
            "document_analysis": document_analysis,
            "query_keywords_used": query_keywords,
            "mode_patterns_used": mode_patterns
        }

    def _calculate_document_relevance(self, doc: Dict, query_keywords: List[str],
                                      mode_patterns: List[str], query_mode: str) -> Dict[str, Any]:
        """Calculate relevance score for a single document"""

        content = doc.get("content", "").lower()
        title = doc.get("metadata", {}).get("title", "").lower()

        # Check keyword matches
        keyword_matches = []
        for keyword in query_keywords:
            if keyword.lower() in content or keyword.lower() in title:
                keyword_matches.append(keyword)

        keyword_coverage = len(keyword_matches) / len(query_keywords) if query_keywords else 0.0

        # Check mode-specific pattern matches
        pattern_matches = []
        for pattern in mode_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                pattern_matches.append(pattern)

        pattern_coverage = len(pattern_matches) / len(mode_patterns) if mode_patterns else 1.0

        # Check automotive context
        automotive_score = self._calculate_automotive_relevance(content)

        # Check content quality indicators
        quality_indicators = {
            "has_specifications": any(spec in content for spec in ["mpg", "horsepower", "torque", "price"]),
            "has_detailed_content": len(content) > 500,
            "has_structured_info": any(marker in content for marker in [":", "-", "•", "specifications", "features"]),
            "recent_content": self._is_recent_content(doc.get("metadata", {})),
        }

        quality_score = sum(quality_indicators.values()) / len(quality_indicators)

        # Calculate overall relevance score with weights
        relevance_score = (
                keyword_coverage * 0.4 +
                pattern_coverage * 0.25 +
                automotive_score * 0.2 +
                quality_score * 0.15
        )

        return {
            "score": min(1.0, relevance_score),
            "matched_keywords": keyword_matches,
            "mode_pattern_matches": pattern_matches,
            "automotive_relevance": automotive_score,
            "quality_indicators": quality_indicators,
            "factors": {
                "keyword_coverage": keyword_coverage,
                "pattern_coverage": pattern_coverage,
                "automotive_score": automotive_score,
                "quality_score": quality_score
            }
        }

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query"""

        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "what", "how", "why", "when", "where", "which", "who", "that", "this",
            "these", "those", "i", "you", "he", "she", "it", "we", "they", "me",
            "him", "her", "us", "them", "my", "your", "his", "her", "its", "our",
            "their", "about", "between", "into", "through", "during", "before",
            "after", "above", "below", "up", "down", "out", "off", "over", "under"
        }

        # Extract words, filter stop words and short words
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return keywords

    def _calculate_automotive_relevance(self, content: str) -> float:
        """Calculate how automotive-focused the content is"""

        automotive_matches = 0
        total_keywords = 0

        for category, keywords in self.automotive_keywords.items():
            category_matches = sum(1 for keyword in keywords if keyword in content)
            automotive_matches += category_matches
            total_keywords += len(keywords)

        # Also check for specific automotive context
        automotive_context_indicators = [
            "vehicle", "automobile", "automotive", "driving", "driver", "road",
            "miles", "kilometers", "dealership", "manufacturer", "model year"
        ]

        context_matches = sum(1 for indicator in automotive_context_indicators if indicator in content)

        # Combine keyword matches and context indicators
        keyword_score = min(1.0, automotive_matches / 20)  # Normalize to reasonable threshold
        context_score = min(1.0, context_matches / len(automotive_context_indicators))

        return (keyword_score * 0.7) + (context_score * 0.3)

    def _is_recent_content(self, metadata: Dict[str, Any]) -> bool:
        """Check if content appears to be recent"""

        # Check for publication date in metadata
        pub_date = metadata.get("publishedDate", metadata.get("date", ""))
        if pub_date:
            try:
                from datetime import datetime
                # Simple check for recent dates (within last 3 years)
                if "2022" in pub_date or "2023" in pub_date or "2024" in pub_date:
                    return True
            except:
                pass

        # Check for recent model years in content
        recent_years = ["2022", "2023", "2024", "2025"]
        url = metadata.get("url", "")
        title = metadata.get("title", "")

        return any(year in url or year in title for year in recent_years)

    def _analyze_query_coverage(self, documents: List[Dict], query: str, query_mode: str) -> Dict[str, Any]:
        """Analyze how well documents cover different aspects of the query"""

        # Extract query aspects based on mode
        query_aspects = self._extract_query_aspects(query, query_mode)

        # Check coverage for each aspect
        aspect_coverage = {}
        for aspect in query_aspects:
            covering_docs = []
            for i, doc in enumerate(documents):
                if self._document_covers_aspect(doc, aspect):
                    covering_docs.append(i)

            aspect_coverage[aspect] = {
                "covering_documents": covering_docs,
                "coverage_count": len(covering_docs),
                "coverage_ratio": len(covering_docs) / len(documents) if documents else 0
            }

        # Calculate overall coverage score
        covered_aspects = sum(1 for aspect_info in aspect_coverage.values() if aspect_info["coverage_count"] > 0)
        coverage_score = covered_aspects / len(query_aspects) if query_aspects else 1.0

        # Identify gaps
        uncovered_aspects = [
            aspect for aspect, info in aspect_coverage.items()
            if info["coverage_count"] == 0
        ]

        poorly_covered_aspects = [
            aspect for aspect, info in aspect_coverage.items()
            if 0 < info["coverage_count"] < 2  # Less than 2 documents
        ]

        return {
            "coverage_score": coverage_score,
            "query_aspects": query_aspects,
            "aspect_coverage": aspect_coverage,
            "covered_aspects": covered_aspects,
            "uncovered_aspects": uncovered_aspects,
            "poorly_covered_aspects": poorly_covered_aspects,
            "coverage_distribution": self._get_coverage_distribution(aspect_coverage)
        }

    def _extract_query_aspects(self, query: str, query_mode: str) -> List[str]:
        """Extract key aspects that should be covered based on the query"""

        aspects = []
        query_lower = query.lower()

        # Mode-specific aspect extraction
        if query_mode == "facts":
            fact_aspects = ["specifications", "performance", "fuel_economy", "pricing", "dimensions"]
            aspects.extend([aspect for aspect in fact_aspects if any(keyword in query_lower for keyword in [
                "spec", "mpg", "hp", "horsepower", "price", "cost", "size", "weight"
            ])])

        elif query_mode == "features":
            feature_aspects = ["technology", "safety_features", "comfort", "convenience", "infotainment"]
            aspects.extend(feature_aspects)

        elif query_mode == "tradeoffs":
            tradeoff_aspects = ["advantages", "disadvantages", "comparisons", "alternatives"]
            aspects.extend(tradeoff_aspects)

        elif query_mode == "scenarios":
            scenario_aspects = ["use_cases", "suitability", "recommendations", "context"]
            aspects.extend(scenario_aspects)

        elif query_mode == "debate":
            debate_aspects = ["expert_opinions", "different_perspectives", "evidence", "analysis"]
            aspects.extend(debate_aspects)

        elif query_mode == "quotes":
            quote_aspects = ["user_experiences", "real_world_feedback", "owner_reviews", "personal_stories"]
            aspects.extend(quote_aspects)

        # Add general automotive aspects if specific query content detected
        general_aspects = []
        if any(keyword in query_lower for keyword in ["reliability", "maintenance", "service"]):
            general_aspects.append("reliability")
        if any(keyword in query_lower for keyword in ["safety", "crash", "rating"]):
            general_aspects.append("safety")
        if any(keyword in query_lower for keyword in ["fuel", "gas", "mpg", "economy"]):
            general_aspects.append("fuel_economy")

        aspects.extend(general_aspects)

        # If no specific aspects found, add default ones
        if not aspects:
            aspects = ["general_information", "basic_details"]

        return list(set(aspects))  # Remove duplicates

    def _document_covers_aspect(self, doc: Dict[str, Any], aspect: str) -> bool:
        """Check if a document covers a specific aspect"""

        content = doc.get("content", "").lower()

        aspect_keywords = {
            "specifications": ["specification", "specs", "technical", "engine", "transmission", "mpg"],
            "performance": ["performance", "horsepower", "torque", "acceleration", "speed", "handling"],
            "fuel_economy": ["mpg", "fuel economy", "gas mileage", "efficiency", "consumption"],
            "pricing": ["price", "cost", "msrp", "expensive", "affordable", "value", "$"],
            "dimensions": ["dimensions", "size", "length", "width", "height", "weight", "cargo"],
            "technology": ["technology", "infotainment", "connectivity", "apps", "screen", "navigation"],
            "safety_features": ["safety", "airbag", "collision", "assist", "monitoring", "nhtsa", "iihs"],
            "comfort": ["comfort", "seats", "ride", "quiet", "smooth", "spacious", "ergonomic"],
            "convenience": ["convenience", "storage", "cup holder", "usb", "charging", "remote"],
            "infotainment": ["infotainment", "entertainment", "audio", "radio", "bluetooth", "android", "apple"],
            "advantages": ["pros", "advantages", "benefits", "positive", "strengths", "good"],
            "disadvantages": ["cons", "disadvantages", "drawbacks", "negative", "weaknesses", "bad"],
            "comparisons": ["vs", "versus", "compared", "comparison", "against", "better", "worse"],
            "alternatives": ["alternative", "option", "choice", "instead", "rather", "other"],
            "use_cases": ["use case", "purpose", "scenario", "situation", "ideal for", "best for"],
            "suitability": ["suitable", "appropriate", "fit", "match", "right for", "works for"],
            "recommendations": ["recommend", "suggest", "advice", "should", "consider", "choose"],
            "context": ["context", "situation", "circumstance", "condition", "environment"],
            "expert_opinions": ["expert", "professional", "analyst", "specialist", "authority"],
            "different_perspectives": ["perspective", "viewpoint", "opinion", "view", "stance"],
            "evidence": ["evidence", "data", "study", "research", "test", "proof", "statistics"],
            "analysis": ["analysis", "analyze", "examination", "evaluation", "assessment"],
            "user_experiences": ["experience", "user", "owner", "customer", "personal"],
            "real_world_feedback": ["real world", "actual", "practical", "everyday", "long term"],
            "owner_reviews": ["owner", "review", "feedback", "rating", "satisfaction"],
            "personal_stories": ["story", "experience", "personal", "my", "i", "journey"],
            "reliability": ["reliability", "dependable", "durable", "problems", "issues", "breakdown"],
            "safety": ["safety", "safe", "crash", "protection", "secure", "rating"],
            "general_information": ["information", "details", "about", "overview", "summary"],
            "basic_details": ["basic", "fundamental", "essential", "key", "important"]
        }

        keywords = aspect_keywords.get(aspect, [aspect])
        return any(keyword in content for keyword in keywords)

    def _get_coverage_distribution(self, aspect_coverage: Dict[str, Any]) -> Dict[str, int]:
        """Get distribution of how many documents cover each aspect"""

        distribution = {"well_covered": 0, "partial_coverage": 0, "not_covered": 0}

        for aspect_info in aspect_coverage.values():
            count = aspect_info["coverage_count"]
            if count >= 3:
                distribution["well_covered"] += 1
            elif count > 0:
                distribution["partial_coverage"] += 1
            else:
                distribution["not_covered"] += 1

        return distribution

    def _analyze_retrieval_diversity(self, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze diversity of retrieved documents"""

        # Analyze source diversity
        source_domains = set()
        source_types = []

        for doc in documents:
            metadata = doc.get("metadata", {})

            # Extract domain
            url = metadata.get("url", "")
            if url:
                domain = self._extract_domain(url)
                if domain:
                    source_domains.add(domain)

            # Classify source type
            source_type = self._classify_source_type(metadata)
            source_types.append(source_type)

        # Calculate diversity metrics
        domain_diversity = len(source_domains) / len(documents) if documents else 0
        type_diversity = len(set(source_types)) / len(source_types) if source_types else 0

        # Analyze content diversity (length, structure, focus)
        content_lengths = [len(doc.get("content", "")) for doc in documents]
        length_variance = self._calculate_variance(content_lengths)

        # Normalize length variance
        max_length = max(content_lengths) if content_lengths else 1
        normalized_length_variance = min(1.0, length_variance / (max_length ** 2))

        # Overall diversity score
        diversity_score = (domain_diversity * 0.4) + (type_diversity * 0.4) + (normalized_length_variance * 0.2)

        return {
            "diversity_score": diversity_score,
            "domain_diversity": domain_diversity,
            "type_diversity": type_diversity,
            "content_length_variance": normalized_length_variance,
            "unique_domains": len(source_domains),
            "source_type_distribution": Counter(source_types),
            "content_length_stats": {
                "min": min(content_lengths) if content_lengths else 0,
                "max": max(content_lengths) if content_lengths else 0,
                "avg": sum(content_lengths) / len(content_lengths) if content_lengths else 0
            }
        }

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""

        if not url:
            return ""

        import re
        match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        return match.group(1).lower() if match else ""

    def _classify_source_type(self, metadata: Dict[str, Any]) -> str:
        """Classify the type of source"""

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
        elif any(domain in url for domain in ['.edu', 'sae.org']):
            return "academic"
        else:
            return "other"

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""

        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((val - mean) ** 2 for val in values) / len(values)
        return variance

    def _analyze_automotive_focus(self, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze how automotive-focused the retrieved documents are"""

        automotive_scores = []

        for doc in documents:
            content = doc.get("content", "")
            automotive_score = self._calculate_automotive_relevance(content)
            automotive_scores.append(automotive_score)

        avg_automotive_score = sum(automotive_scores) / len(automotive_scores) if automotive_scores else 0.0

        high_automotive_docs = sum(1 for score in automotive_scores if score >= 0.8)
        low_automotive_docs = sum(1 for score in automotive_scores if score < 0.5)

        return {
            "automotive_focus_score": avg_automotive_score,
            "automotive_distribution": {
                "high": high_automotive_docs,
                "medium": len(automotive_scores) - high_automotive_docs - low_automotive_docs,
                "low": low_automotive_docs
            },
            "document_scores": automotive_scores
        }

    def _analyze_quality_distribution(self, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze the quality distribution of retrieved documents"""

        quality_scores = []

        for doc in documents:
            quality_score = self._assess_document_quality(doc)
            quality_scores.append(quality_score)

        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        high_quality_docs = sum(1 for score in quality_scores if score >= 0.8)
        low_quality_docs = sum(1 for score in quality_scores if score < 0.5)

        return {
            "average_quality": avg_quality,
            "quality_distribution": {
                "high": high_quality_docs,
                "medium": len(quality_scores) - high_quality_docs - low_quality_docs,
                "low": low_quality_docs
            },
            "document_quality_scores": quality_scores
        }

    def _assess_document_quality(self, doc: Dict[str, Any]) -> float:
        """Assess the quality of a single document"""

        content = doc.get("content", "")
        metadata = doc.get("metadata", {})

        quality_factors = {
            "content_length": min(1.0, len(content) / 1000),  # Longer content often higher quality
            "has_title": 1.0 if metadata.get("title") else 0.0,
            "has_author": 1.0 if metadata.get("author") else 0.0,
            "has_date": 1.0 if metadata.get("publishedDate") else 0.0,
            "structured_content": 1.0 if self._has_structured_content(content) else 0.0,
            "has_specific_data": 1.0 if self._has_specific_automotive_data(content) else 0.0
        }

        # Weight the factors
        weights = {
            "content_length": 0.2,
            "has_title": 0.15,
            "has_author": 0.1,
            "has_date": 0.1,
            "structured_content": 0.25,
            "has_specific_data": 0.2
        }

        quality_score = sum(quality_factors[factor] * weights[factor] for factor in quality_factors)

        return min(1.0, quality_score)

    def _has_structured_content(self, content: str) -> bool:
        """Check if content appears to be well-structured"""

        structure_indicators = [
            ":", "-", "•", "1.", "2.", "specifications", "features",
            "pros", "cons", "advantages", "disadvantages"
        ]

        indicator_count = sum(1 for indicator in structure_indicators if indicator in content)
        return indicator_count >= 3

    def _has_specific_automotive_data(self, content: str) -> bool:
        """Check if content contains specific automotive data"""

        data_indicators = [
            "mpg", "horsepower", "torque", "$", "hp", "lb-ft",
            "0-60", "top speed", "weight", "length", "width"
        ]

        return any(indicator in content.lower() for indicator in data_indicators)

    def _assess_retrieval_effectiveness(self, quality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall retrieval effectiveness"""

        relevance_analysis = quality_analysis["relevance_analysis"]
        coverage_analysis = quality_analysis["coverage_analysis"]
        diversity_analysis = quality_analysis["diversity_analysis"]

        effectiveness_metrics = {
            "precision": relevance_analysis["average_relevance"],  # How relevant are retrieved docs
            "recall_proxy": coverage_analysis["coverage_score"],  # How well query is covered
            "diversity": diversity_analysis["diversity_score"],  # How diverse are sources
            "quality": quality_analysis["quality_distribution"]["average_quality"]
        }

        # Calculate F1-like score for precision and recall proxy
        precision = effectiveness_metrics["precision"]
        recall_proxy = effectiveness_metrics["recall_proxy"]

        if precision + recall_proxy > 0:
            f1_score = 2 * (precision * recall_proxy) / (precision + recall_proxy)
        else:
            f1_score = 0.0

        return {
            "effectiveness_metrics": effectiveness_metrics,
            "f1_score": f1_score,
            "overall_effectiveness": f1_score * 0.6 + effectiveness_metrics["diversity"] * 0.2 + effectiveness_metrics[
                "quality"] * 0.2
        }

    def _generate_retrieval_improvements(self, quality_analysis: Dict[str, Any], query_mode: str) -> List[str]:
        """Generate specific suggestions for improving retrieval"""

        suggestions = []

        # Relevance improvements
        relevance_analysis = quality_analysis["relevance_analysis"]
        if relevance_analysis["average_relevance"] < 0.7:
            suggestions.append("Improve query terms to better match automotive content")
            if relevance_analysis["relevance_distribution"]["low"] > 2:
                suggestions.append("Filter out low-relevance documents before analysis")

        # Coverage improvements
        coverage_analysis = quality_analysis["coverage_analysis"]
        if coverage_analysis["uncovered_aspects"]:
            uncovered = coverage_analysis["uncovered_aspects"][:3]  # Top 3
            suggestions.append(f"Add documents covering: {', '.join(uncovered)}")

        # Diversity improvements
        diversity_analysis = quality_analysis["diversity_analysis"]
        if diversity_analysis["diversity_score"] < 0.6:
            type_dist = diversity_analysis["source_type_distribution"]
            if len(type_dist) < 3:
                suggestions.append("Include sources from more diverse types (official, professional, user reviews)")
            if diversity_analysis["unique_domains"] < 3:
                suggestions.append("Add sources from different websites/domains")

        # Automotive focus improvements
        automotive_analysis = quality_analysis["automotive_focus_analysis"]
        if automotive_analysis["automotive_focus_score"] < 0.8:
            suggestions.append("Focus retrieval on automotive-specific content")

        # Quality improvements
        quality_analysis_dist = quality_analysis["quality_distribution"]
        if quality_analysis_dist["quality_distribution"]["low"] > 1:
            suggestions.append("Improve document quality by favoring detailed, well-structured sources")

        return suggestions

    def _create_unverifiable_result(self, start_time: datetime,
                                    precondition_result: PreconditionResult) -> ValidationStepResult:
        """Create result for unverifiable validation"""

        return ValidationStepResult(
            step_id=f"retrieval_quality_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Retrieval Quality Analysis",
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
                    message="Cannot analyze retrieval quality",
                    explanation=precondition_result.failure_reason,
                    suggestion="Ensure documents are retrieved successfully"
                )
            ]
        )

    def _create_error_result(self, start_time: datetime, error_message: str) -> ValidationStepResult:
        """Create result for validation errors"""

        return ValidationStepResult(
            step_id=f"retrieval_quality_{datetime.now().isoformat()}",
            step_type=self.step_type,
            step_name="Retrieval Quality Analysis",
            status=ValidationStatus.FAILED,
            confidence_impact=-8.0,
            summary=f"Analysis failed: {error_message}",
            details={"error": error_message},
            started_at=start_time,
            completed_at=datetime.now(),
            warnings=[
                ValidationWarning(
                    category="validation_error",
                    severity="critical",
                    message="Retrieval quality analysis encountered an error",
                    explanation=error_message,
                    suggestion="Check logs and retry analysis"
                )
            ]
        )