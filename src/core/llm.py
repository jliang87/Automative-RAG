import json
import time
import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from src.config.settings import settings
from src.core.mode_config import mode_config, QueryMode

# Import ALL shared utilities - NO DUPLICATION
from src.utils.quality_utils import (
    extract_automotive_key_phrases,
    check_acceleration_claims,
    check_numerical_specs_realistic,
    has_numerical_data,
    extract_key_terms,
    has_garbled_content
)

logger = logging.getLogger(__name__)


def _format_documents_for_context(
        documents: List[Tuple[Document, float]],
        max_token_budget: Optional[int] = None,
        include_relevance_scores: bool = True
) -> str:
    """
    Format retrieved documents into context for the prompt with relevance scores and token management.

    ENHANCED:
    - Shows relevance scores to help LLM assess trustworthiness
    - Respects token budget limits
    - Prioritizes higher-scoring documents

    Args:
        documents: List of (document, relevance_score) tuples
        max_token_budget: Maximum tokens to use for context (optional)
        include_relevance_scores: Whether to show relevance scores to LLM

    Returns:
        Formatted context string
    """
    from src.core.mode_config import estimate_token_count

    if not documents:
        return "No relevant documents found."

    context_parts = []
    total_tokens = 0

    # Sort by relevance score (highest first) to prioritize best content
    sorted_documents = sorted(documents, key=lambda x: x[1], reverse=True)

    for i, (doc, score) in enumerate(sorted_documents):
        # Extract metadata for citation
        metadata = doc.metadata
        source_type = metadata.get("source", "unknown")
        title = metadata.get("title", f"Document {i + 1}")

        # Format source information with unique ID
        doc_id = f"DOC_{i + 1}"
        if source_type == "youtube":
            source_info = f"{doc_id} (YouTube - '{title}'"
            if "url" in metadata:
                source_info += f" - {metadata['url']}"
            source_info += ")"
        elif source_type == "bilibili":
            source_info = f"{doc_id} (Bilibili - '{title}'"
            if "url" in metadata:
                source_info += f" - {metadata['url']}"
            source_info += ")"
        elif source_type == "pdf":
            source_info = f"{doc_id} (PDF - '{title}')"
        else:
            source_info = f"{doc_id} ({title})"

        # Add manufacturer and model if available
        manufacturer = metadata.get("manufacturer")
        model = metadata.get("model")
        year = metadata.get("year")

        if manufacturer or model or year:
            source_info += " - "
            if manufacturer:
                source_info += manufacturer
            if model:
                source_info += f" {model}"
            if year:
                source_info += f" ({year})"

        # ENHANCED: Add relevance score to help LLM assess trustworthiness
        if include_relevance_scores:
            confidence_indicator = "🔥" if score > 0.8 else "⭐" if score > 0.6 else "📄"
            source_info += f" {confidence_indicator} (Relevance: {score:.2f})"

        # Create content block
        content_block = f"{source_info}\n{doc.page_content}\n"

        # ENHANCED: Token budget management
        if max_token_budget:
            block_tokens = estimate_token_count(content_block)

            # Check if adding this block would exceed budget
            if total_tokens + block_tokens > max_token_budget:
                # Try to include a truncated version if this is important content
                if score > 0.7 and total_tokens < max_token_budget * 0.8:
                    # Calculate how much content we can include
                    remaining_tokens = max_token_budget - total_tokens - estimate_token_count(source_info + "\n")
                    remaining_chars = int(remaining_tokens * 2.5)  # Rough char-to-token ratio

                    if remaining_chars > 100:  # Only if we can include meaningful content
                        truncated_content = doc.page_content[:remaining_chars] + "... [截断]"
                        content_block = f"{source_info}\n{truncated_content}\n"
                        context_parts.append(content_block)
                        total_tokens = max_token_budget  # We're at the limit
                        logger.info(f"Truncated high-relevance document {doc_id} to fit token budget")

                break  # Stop adding more documents

            total_tokens += block_tokens

        context_parts.append(content_block)

        # Safety limit to prevent extremely long contexts
        if len(context_parts) >= 20:  # Maximum 20 documents
            logger.info("Reached maximum document limit (20), stopping context building")
            break

    final_context = "\n\n".join(context_parts)

    # Log context statistics
    final_tokens = estimate_token_count(final_context)
    logger.info(f"Context built: {len(context_parts)} documents, ~{final_tokens} tokens")
    if max_token_budget:
        logger.info(
            f"Token budget utilization: {final_tokens}/{max_token_budget} ({(final_tokens / max_token_budget) * 100:.1f}%)")

    return final_context


def get_context_with_token_budget(
        documents: List[Tuple[Document, float]],
        query_mode: str = "facts",
        custom_budget: Optional[int] = None
) -> str:
    """
    Convenience function to get context with mode-specific token budget.

    Args:
        documents: List of (document, relevance_score) tuples
        query_mode: Query mode to determine token budget
        custom_budget: Override the mode-specific budget

    Returns:
        Formatted context string within token limits
    """
    try:
        mode_enum = QueryMode(query_mode)
    except ValueError:
        mode_enum = QueryMode.FACTS

    # Get mode-specific token budget
    context_params = mode_config.get_context_params(mode_enum)
    token_budget = custom_budget or context_params["max_context_tokens"]

    return _format_documents_for_context(
        documents=documents,
        max_token_budget=token_budget,
        include_relevance_scores=True
    )


class ContradictionDetector:
    """
    ENHANCED: Detect contradictions in numerical specifications across documents.
    """

    def __init__(self):
        # Define key automotive specs to check for contradictions
        self.spec_patterns = {
            "acceleration": [
                r'(?:百公里加速|0-100|零至100|加速时间).*?(\d+\.?\d*)\s*秒',
                r'(\d+\.?\d*)\s*秒.*?(?:百公里加速|0-100|加速时间)'
            ],
            "top_speed": [
                r'(?:最高时速|最大速度|极速|顶速).*?(\d+)\s*(?:公里|千米|km/h)',
                r'时速.*?(?:最高|最大|可达).*?(\d+)\s*(?:公里|千米|km/h)'
            ],
            "horsepower": [
                r'(?:最大功率|功率|马力).*?(\d+)\s*(?:马力|HP|hp)',
                r'(\d+)\s*(?:马力|HP|hp).*?(?:功率|输出|最大)'
            ],
            "trunk_capacity": [
                r'(?:后备箱|行李箱|尾箱).*?(?:容积|空间).*?(\d+)\s*(?:升|L)',
                r'(?:容积|空间).*?(\d+)\s*(?:升|L).*?(?:后备箱|行李箱)'
            ],
            "fuel_consumption": [
                r'(?:油耗|燃油消耗|百公里油耗).*?(\d+\.?\d*)\s*(?:升|L)',
                r'(\d+\.?\d*)\s*(?:升|L).*?(?:百公里|油耗)'
            ]
        }

    def detect_contradictions(self, documents: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """
        Detect contradictions in specifications across documents.

        Returns:
            Dictionary with detected contradictions and recommendations
        """
        contradictions = {}

        for spec_name, patterns in self.spec_patterns.items():
            spec_values = []

            # Extract values for this spec from all documents
            for i, (doc, score) in enumerate(documents):
                content = doc.page_content
                doc_id = f"DOC_{i + 1}"
                doc_title = doc.metadata.get("title", f"Document {i + 1}")

                for pattern in patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        try:
                            value = float(match)
                            spec_values.append({
                                "value": value,
                                "doc_id": doc_id,
                                "doc_title": doc_title,
                                "source": doc.metadata.get("source", "unknown")
                            })
                        except ValueError:
                            continue

            # Check for contradictions (significant differences)
            if len(spec_values) > 1:
                values = [item["value"] for item in spec_values]
                min_val, max_val = min(values), max(values)

                # Consider it a contradiction if difference is > 20% or > 10 for acceleration
                if spec_name == "acceleration":
                    threshold_diff = 2.0  # 2 seconds difference
                else:
                    threshold_diff = min_val * 0.2  # 20% difference

                if (max_val - min_val) > threshold_diff:
                    contradictions[spec_name] = {
                        "spec_name_chinese": self._get_chinese_spec_name(spec_name),
                        "conflicting_values": spec_values,
                        "min_value": min_val,
                        "max_value": max_val,
                        "difference": max_val - min_val,
                        "recommendation": self._generate_contradiction_advice(spec_name, spec_values)
                    }

        return contradictions

    def _get_chinese_spec_name(self, spec_name: str) -> str:
        """Get Chinese name for specification."""
        chinese_names = {
            "acceleration": "百公里加速",
            "top_speed": "最高时速",
            "horsepower": "马力",
            "trunk_capacity": "后备箱容积",
            "fuel_consumption": "油耗"
        }
        return chinese_names.get(spec_name, spec_name)

    def _generate_contradiction_advice(self, spec_name: str, values: List[Dict]) -> str:
        """Generate advice for handling contradictions."""
        sources = [item["source"] for item in values]

        if "youtube" in sources and "bilibili" in sources:
            return "建议优先参考视频测试数据，并核实测试条件是否一致"
        elif "pdf" in sources:
            return "建议优先参考官方PDF文档数据"
        else:
            return "建议进一步核实数据来源的权威性"


class StructuredOutputGenerator:
    """
    ENHANCED: Generate structured JSON output to reduce hallucination.
    """

    def __init__(self):
        self.structured_fields = {
            "basic_specs": {
                "acceleration_0_100_kph": "百公里加速时间（秒）",
                "top_speed_kph": "最高时速（公里/小时）",
                "horsepower": "马力",
                "trunk_capacity_liters": "后备箱容积（升）",
                "fuel_consumption_l_100km": "油耗（升/百公里）"
            },
            "dimensions": {
                "length_mm": "车长（毫米）",
                "width_mm": "车宽（毫米）",
                "height_mm": "车高（毫米）",
                "wheelbase_mm": "轴距（毫米）"
            },
            "powertrain": {
                "engine_type": "发动机类型",
                "transmission": "变速箱",
                "drivetrain": "驱动方式"
            }
        }

    def extract_structured_data(self, answer: str, documents: List[Tuple[Document, float]]) -> Dict[str, Any]:
        """
        Extract structured data from answer and documents.

        Returns:
            Structured JSON representation of automotive specifications
        """
        structured_data = {
            "basic_specs": {},
            "dimensions": {},
            "powertrain": {},
            "sources": [],
            "extraction_confidence": "medium"
        }

        # Extract sources
        for i, (doc, score) in enumerate(documents):
            source_info = {
                "id": f"DOC_{i + 1}",
                "title": doc.metadata.get("title", ""),
                "source_type": doc.metadata.get("source", "unknown"),
                "url": doc.metadata.get("url", ""),
                "relevance_score": float(score)
            }
            structured_data["sources"].append(source_info)

        # Extract numerical specifications using regex patterns
        self._extract_basic_specs(answer, structured_data)
        self._extract_dimensions(answer, structured_data)
        self._extract_powertrain(answer, structured_data)

        # Calculate extraction confidence
        filled_fields = sum(
            len([v for v in category.values() if v])
            for category in
            [structured_data["basic_specs"], structured_data["dimensions"], structured_data["powertrain"]]
        )

        total_fields = sum(len(category) for category in self.structured_fields.values())
        confidence_ratio = filled_fields / total_fields

        if confidence_ratio > 0.7:
            structured_data["extraction_confidence"] = "high"
        elif confidence_ratio > 0.3:
            structured_data["extraction_confidence"] = "medium"
        else:
            structured_data["extraction_confidence"] = "low"

        return structured_data

    def _extract_basic_specs(self, text: str, data: Dict):
        """Extract basic specifications from text."""
        # Acceleration
        acc_patterns = [
            r'(?:百公里加速|0-100|加速时间).*?(\d+\.?\d*)\s*秒',
            r'(\d+\.?\d*)\s*秒.*?(?:百公里加速|加速)'
        ]
        for pattern in acc_patterns:
            match = re.search(pattern, text)
            if match:
                data["basic_specs"]["acceleration_0_100_kph"] = float(match.group(1))
                break

        # Top speed
        speed_patterns = [
            r'(?:最高时速|极速|顶速).*?(\d+)\s*(?:公里|km/h)',
            r'时速.*?(\d+)\s*(?:公里|km/h)'
        ]
        for pattern in speed_patterns:
            match = re.search(pattern, text)
            if match:
                data["basic_specs"]["top_speed_kph"] = int(match.group(1))
                break

        # Horsepower
        hp_patterns = [
            r'(?:马力|功率).*?(\d+)\s*(?:马力|HP|hp)',
            r'(\d+)\s*(?:马力|HP|hp)'
        ]
        for pattern in hp_patterns:
            match = re.search(pattern, text)
            if match:
                data["basic_specs"]["horsepower"] = int(match.group(1))
                break

    def _extract_dimensions(self, text: str, data: Dict):
        """Extract dimension specifications from text."""
        # Length, width, height patterns
        dim_patterns = {
            "length_mm": [r'车长.*?(\d{4,5})\s*(?:毫米|mm)', r'长度.*?(\d{4,5})'],
            "width_mm": [r'车宽.*?(\d{4,5})\s*(?:毫米|mm)', r'宽度.*?(\d{4,5})'],
            "height_mm": [r'车高.*?(\d{4,5})\s*(?:毫米|mm)', r'高度.*?(\d{4,5})'],
            "wheelbase_mm": [r'轴距.*?(\d{4,5})\s*(?:毫米|mm)']
        }

        for field, patterns in dim_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    data["dimensions"][field] = int(match.group(1))
                    break

    def _extract_powertrain(self, text: str, data: Dict):
        """Extract powertrain information from text."""
        # Engine type
        if re.search(r'汽油|gasoline|petrol', text, re.IGNORECASE):
            data["powertrain"]["engine_type"] = "汽油"
        elif re.search(r'柴油|diesel', text, re.IGNORECASE):
            data["powertrain"]["engine_type"] = "柴油"
        elif re.search(r'电动|electric|EV|纯电', text, re.IGNORECASE):
            data["powertrain"]["engine_type"] = "电动"
        elif re.search(r'混合动力|hybrid|混动', text, re.IGNORECASE):
            data["powertrain"]["engine_type"] = "混合动力"

        # Transmission
        if re.search(r'自动|automatic|自动挡', text, re.IGNORECASE):
            data["powertrain"]["transmission"] = "自动"
        elif re.search(r'手动|manual|手动挡', text, re.IGNORECASE):
            data["powertrain"]["transmission"] = "手动"
        elif re.search(r'CVT|无级变速', text, re.IGNORECASE):
            data["powertrain"]["transmission"] = "CVT"


class AutomotiveFactChecker:
    """
    Fact checker for automotive specifications to detect obvious hallucinations.
    ENHANCED: Now includes contradiction detection.
    """

    def __init__(self):
        # Define realistic ranges for automotive specs
        self.spec_ranges = {
            "acceleration_0_100": (2.0, 20.0),  # 0-100 km/h in seconds
            "top_speed": (120, 400),  # km/h
            "horsepower": (50, 2000),  # HP
            "torque": (50, 2000),  # Nm
            "fuel_consumption": (3.0, 25.0),  # L/100km
            "engine_displacement": (0.5, 8.0),  # Liters
            "trunk_capacity": (100, 2000),  # Liters
            "wheelbase": (2000, 4000),  # mm
            "length": (3000, 7000),  # mm
            "width": (1500, 2500),  # mm
            "height": (1200, 2500),  # mm
            "weight": (800, 5000),  # kg
            "price": (50000, 5000000),  # CNY
        }

        # ENHANCED: Add contradiction detector
        self.contradiction_detector = ContradictionDetector()

    def check_answer_quality(self, answer: str, context: str, documents: List[Tuple[Document, float]] = None) -> Dict[
        str, any]:
        """
        Comprehensive answer quality check using shared utility functions.
        ENHANCED: Now includes contradiction detection.
        """
        warnings = []

        # Use shared utility functions - NO DUPLICATION
        warnings.extend(check_acceleration_claims(answer))
        warnings.extend(check_numerical_specs_realistic(answer))
        warnings.extend(self._verify_context_support(answer, context))

        # ENHANCED: Check for contradictions in source documents
        contradictions = {}
        if documents:
            contradictions = self.contradiction_detector.detect_contradictions(documents)
            if contradictions:
                for spec_name, contradiction_info in contradictions.items():
                    spec_chinese = contradiction_info["spec_name_chinese"]
                    warning = f"⚠️ 发现{spec_chinese}数据冲突: {contradiction_info['recommendation']}"
                    warnings.append(warning)

        # Calculate quality score
        quality_score = max(0, 100 - len(warnings) * 15)

        return {
            "warnings": warnings,
            "quality_score": quality_score,
            "has_issues": len(warnings) > 0,
            "contradictions": contradictions,
            "recommendation": "review_answer" if len(warnings) > 2 else "acceptable"
        }

    def _verify_context_support(self, answer: str, context: str) -> List[str]:
        """Check if numerical claims in answer are supported by context (Chinese warnings)."""
        warnings = []

        # Extract numbers from answer
        answer_numbers = re.findall(r'\d+\.?\d*', answer)

        # Check if these numbers exist in context
        for number in answer_numbers:
            if number not in context:
                warnings.append(f"⚠️ 答案中的数字 '{number}' 在提供的文档中未找到")

        return warnings


class AnswerConfidenceScorer:
    """
    Calculate confidence scores for generated answers to help detect potential hallucinations.
    DEDUPED: Uses shared utility functions from quality_utils.py
    """

    def __init__(self):
        self.fact_checker = AutomotiveFactChecker()

    def calculate_confidence(self, answer: str, context: str, documents: List[Tuple[Document, float]]) -> Dict[
        str, any]:
        """
        Calculate comprehensive confidence score for an answer.
        ENHANCED: Now includes contradiction analysis.
        """
        scores = {}

        # 1. Context Support Score (0-100) - uses quality_utils
        scores['context_support'] = self._calculate_context_support(answer, context)

        # 2. Document Relevance Score (0-100)
        scores['document_relevance'] = self._calculate_document_relevance(answer, documents)

        # 3. Factual Consistency Score (0-100) - uses quality_utils via fact_checker
        scores['factual_consistency'] = self._calculate_factual_consistency(answer, context, documents)

        # 4. Specificity Score (0-100) - uses quality_utils functions
        scores['specificity'] = self._calculate_specificity(answer)

        # 5. Uncertainty Indicators (0-100, higher = more uncertain) - uses quality_utils constants
        scores['uncertainty'] = self._detect_uncertainty_indicators(answer)

        # Calculate overall confidence (weighted average)
        weights = {
            'context_support': 0.35,
            'document_relevance': 0.25,
            'factual_consistency': 0.25,
            'specificity': 0.10,
            'uncertainty': -0.05  # Negative weight for uncertainty
        }

        overall_confidence = sum(scores[key] * weights[key] for key in weights.keys())
        overall_confidence = max(0, min(100, overall_confidence))

        # Generate recommendation
        recommendation = self._generate_recommendation(overall_confidence, scores)

        return {
            'overall_confidence': overall_confidence,
            'detailed_scores': scores,
            'recommendation': recommendation,
            'confidence_level': self._get_confidence_level(overall_confidence),
            'should_flag': overall_confidence < 60
        }

    def _calculate_context_support(self, answer: str, context: str) -> float:
        """Calculate how well the answer is supported by the provided context."""
        if not context.strip():
            return 0.0

        # DEDUPED: Use shared utility function
        answer_phrases = extract_automotive_key_phrases(answer)

        # Check how many phrases are found in context
        supported_phrases = 0
        for phrase in answer_phrases:
            if phrase.lower() in context.lower():
                supported_phrases += 1

        support_ratio = supported_phrases / len(answer_phrases) if answer_phrases else 0
        return support_ratio * 100

    def _calculate_document_relevance(self, answer: str, documents: List[Tuple[Document, float]]) -> float:
        """Calculate relevance based on document similarity scores."""
        if not documents:
            return 0.0

        # Use average similarity score as relevance indicator
        avg_score = sum(score for _, score in documents) / len(documents)

        # Normalize to 0-100 scale (assuming scores are typically 0-1)
        return min(100, avg_score * 100)

    def _calculate_factual_consistency(self, answer: str, context: str,
                                       documents: List[Tuple[Document, float]] = None) -> float:
        """Check for factual consistency using the fact checker."""
        quality_check = self.fact_checker.check_answer_quality(answer, context, documents)

        # Convert quality score to consistency score
        return quality_check['quality_score']

    def _calculate_specificity(self, answer: str) -> float:
        """
        Calculate how specific the answer is - DEDUPED version.
        Uses quality_utils functions instead of duplicating logic.
        """
        specificity_indicators = 0

        # 1. Check for specific numbers
        if re.search(r'\d+\.?\d*', answer):
            specificity_indicators += 1

        # 2. DEDUPED: Use quality_utils to check for numerical data (includes units)
        if has_numerical_data(answer):
            specificity_indicators += 1

        # 3. DEDUPED: Use quality_utils to extract automotive phrases (includes brands)
        automotive_phrases = extract_automotive_key_phrases(answer)
        if automotive_phrases:
            specificity_indicators += 1

        # 4. Check for year mentions (Chinese format)
        if re.search(r'(?:20\d{2}|19\d{2})年?', answer):
            specificity_indicators += 1

        # Normalize to 0-100
        max_indicators = 4
        return (specificity_indicators / max_indicators) * 100

    def _detect_uncertainty_indicators(self, answer: str) -> float:
        """
        Detect uncertainty indicators in Chinese answers.
        CENTRALIZED: Define uncertainty phrases here since they're specific to confidence scoring.
        """
        # Chinese uncertainty indicators (centralized definition)
        uncertainty_phrases = [
            # Core uncertainty words
            '可能', '大概', '估计', '应该', '似乎', '看起来', '据说',
            '大致', '约', '左右', '差不多', '基本上', '一般来说',
            '通常', '可能是', '或许', '也许', '估算', '预计',
            '疑似', '推测', '猜测', '不确定', '不清楚', '不详',
            '可能会', '应该是', '看上去', '听说', '传说',
            # Keep minimal English for edge cases
            'maybe', 'probably', 'likely', 'appears', 'seems'
        ]

        uncertainty_count = 0
        for phrase in uncertainty_phrases:
            uncertainty_count += answer.lower().count(phrase.lower())

        # Normalize to 0-100 (higher = more uncertain)
        max_uncertainty = 5  # If more than 5 uncertainty indicators, max score
        return min(100, (uncertainty_count / max_uncertainty) * 100)

    def _generate_recommendation(self, overall_confidence: float, scores: Dict[str, float]) -> str:
        """Generate actionable recommendation based on confidence scores (in Chinese for users)."""
        if overall_confidence >= 85:
            return "高置信度答案，可以直接使用"
        elif overall_confidence >= 70:
            return "中等置信度，建议进行人工验证"
        elif overall_confidence >= 50:
            return "低置信度，需要额外验证和来源确认"
        else:
            return "极低置信度，可能存在错误，建议重新查询"

    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level label (in Chinese for users)."""
        if confidence >= 85:
            return "高"
        elif confidence >= 70:
            return "中"
        elif confidence >= 50:
            return "低"
        else:
            return "极低"


class LocalLLM:
    """
    Local DeepSeek LLM integration for RAG with GPU acceleration.

    ENHANCED: Now supports sentence-level citations, contradiction detection, and structured output.
    """

    def __init__(
            self,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
    ):
        """
        Initialize the local DeepSeek LLM with environment-driven configuration.
        """

        # Use environment settings as defaults - HOLISTIC APPROACH
        self.model_name = model_name or settings.default_llm_model
        self.model_path = settings.llm_model_full_path

        # Device configuration from environment (Tesla T4 optimized)
        self.device = device or settings.device

        # Generation settings from environment
        self.temperature = temperature or settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

        # Quantization settings from environment - TESLA T4 SAFE DEFAULTS
        self.use_4bit = settings.llm_use_4bit  # Default: false for Tesla T4
        self.use_8bit = settings.llm_use_8bit  # Default: false
        self.torch_dtype = settings.llm_torch_dtype  # Default: float16

        # Initialize enhanced components
        self.fact_checker = AutomotiveFactChecker()
        self.confidence_scorer = AnswerConfidenceScorer()
        self.structured_generator = StructuredOutputGenerator()

        # Log configuration for debugging
        print(f"LocalLLM Configuration (ENHANCED: Advanced Anti-Hallucination):")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Use 4-bit: {self.use_4bit}")
        print(f"  Use 8-bit: {self.use_8bit}")
        print(f"  Torch dtype: {self.torch_dtype}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Template Language: ENGLISH (token efficient)")
        print(f"  Response Language: CHINESE (enforced)")
        print(f"  Enhanced Features: SENTENCE_CITATIONS + CONTRADICTION_DETECTION + STRUCTURED_OUTPUT")

        # Initialize tokenizer and model
        self._load_model()

        # ENHANCED: English templates with sentence-level citation requirements
        self.qa_prompt_template = self._create_enhanced_anti_hallucination_template()

    def _load_model(self):
        """Load the local LLM model using environment-driven Tesla T4 configuration."""
        print(f"Loading LLM model {self.model_path} on {self.device}...")

        # Start timing
        start_time = time.time()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True
        )

        # Apply Tesla T4 memory fraction for GPU workers
        if self.device.startswith("cuda") and torch.cuda.is_available():
            memory_fraction = settings.get_worker_memory_fraction()
            if memory_fraction < 1.0:
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                print(f"Set GPU memory fraction to {memory_fraction} (Tesla T4 optimized)")

        # Get model loading kwargs from settings - ENVIRONMENT DRIVEN
        model_kwargs = settings.get_model_kwargs()

        # Log what we're about to do - CRITICAL FOR TESLA T4 DEBUGGING
        quantization_config = model_kwargs.get("quantization_config")
        if quantization_config:
            if hasattr(quantization_config, 'load_in_4bit') and quantization_config.load_in_4bit:
                print("⚠️ Loading with 4-bit quantization (may cause Tesla T4 issues)")
            elif hasattr(quantization_config, 'load_in_8bit') and quantization_config.load_in_8bit:
                print("Loading with 8-bit quantization")
        else:
            print(f"✅ Loading with {self.torch_dtype} precision (Tesla T4 optimized)")

        try:
            # Load model with environment-driven configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )

            # Move to device if not using device_map
            if not model_kwargs.get("device_map") and self.device != "cpu":
                self.model = self.model.to(self.device)

            # Create generation pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=False,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                repetition_penalty=settings.llm_repetition_penalty
            )

            # Report loading time and memory usage
            load_time = time.time() - start_time
            print(f"✅ Model loaded successfully in {load_time:.2f} seconds")

            if torch.cuda.is_available() and self.device.startswith("cuda"):
                memory_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                memory_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                print(
                    f"GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved ({total_memory:.2f}GB total)")

        except RuntimeError as e:
            if "CUDA driver error: invalid argument" in str(e):
                print("❌ Tesla T4 compatibility error detected!")
                print("This is likely caused by 4-bit quantization on Tesla T4.")
                print("Current settings:")
                print(f"  LLM_USE_4BIT: {settings.llm_use_4bit}")
                print(f"  LLM_USE_8BIT: {settings.llm_use_8bit}")
                print("Solution: Set LLM_USE_4BIT=false in your .env file")
                raise e
            else:
                raise e

    def _create_enhanced_anti_hallucination_template(self) -> str:
        """
        ENHANCED: Create English template with sentence-level citation requirements.
        This dramatically improves citation granularity and reduces hallucination.
        """
        template = """You are a professional automotive specifications expert assistant with STRICT accuracy requirements.

CRITICAL RULES:
1. Only use information explicitly mentioned in the provided documents
2. If specific numbers/specs are not in documents, say "According to provided documents, specific [parameter] data not found"
3. Never estimate, guess, or infer any numerical values
4. If document content is unclear or contradictory, acknowledge this uncertainty
5. MANDATORY: Cite the source document for EVERY factual sentence using the format 【来源：DOC_X】

SENTENCE-LEVEL CITATION REQUIREMENT:
- Every sentence containing facts must end with 【来源：DOC_X (title)】
- Multiple sources can be cited as 【来源：DOC_1, DOC_2】
- Opinion or uncertainty statements don't need citations
- Example: "最高时速为220公里/小时【来源：DOC_1 (Bilibili - 实测比亚迪汉EV极速表现)】。"

NUMERICAL ACCURACY CHECK:
- 0-100 km/h acceleration: normal range is 3-15 seconds
- If you see obviously wrong values (like 0.8 seconds), mark as suspicious
- Always double-check technical specs against automotive standards

CONTRADICTION HANDLING:
- If documents contain conflicting information, mention both values with their sources
- Example: "加速时间为3.9秒【来源：DOC_1】，但另一来源显示为4.2秒【来源：DOC_2】，建议进一步核实。"

Your task is to help users find automotive specifications, features, and technical details.

Use ONLY the following document content to answer questions. Each document has a unique ID (DOC_1, DOC_2, etc.).

Document Content:
{context}

Question:
{question}

IMPORTANT: 
1. Respond in Chinese with precise facts
2. Cite specific document sources for EVERY factual sentence using 【来源：DOC_X】
3. If contradictions exist, present both values with sources
4. If uncertain, clearly state limitations"""
        return template

    def get_prompt_template_for_mode(self, mode: str, structured_output: bool = False) -> str:
        """
        ENHANCED: Get English prompt templates with sentence-level citations and optional structured output.
        """

        # Base citation requirement for all modes
        citation_requirement = """
SENTENCE-LEVEL CITATION REQUIREMENT:
- Every sentence containing facts must end with 【来源：DOC_X (title)】
- Multiple sources can be cited as 【来源：DOC_1, DOC_2】
- Example: "最高时速为220公里/小时【来源：DOC_1 (YouTube - 性能测试视频)】。"
"""

        structured_instruction = ""
        if structured_output:
            structured_instruction = """
STRUCTURED OUTPUT REQUIREMENT:
After providing the natural language answer, also provide a JSON structure with extracted specifications:

```json
{
  "basic_specs": {
    "acceleration_0_100_kph": "数值或null",
    "top_speed_kph": "数值或null", 
    "horsepower": "数值或null"
  },
  "sources": ["DOC_1: 标题", "DOC_2: 标题"]
}
```
"""

        templates = {
            "facts": f"""You are a professional automotive specifications expert assistant with strict accuracy requirements.

CRITICAL RULES:
1. Only use information explicitly mentioned in the provided documents
2. If specific numbers/specs are not in documents, say "According to provided documents, specific [parameter] data not found"
3. Never estimate, guess, or infer any numerical values
4. If document content is unclear or contradictory, acknowledge this uncertainty
5. Always cite the exact sources where information was found

{citation_requirement}

NUMERICAL ACCURACY CHECK:
- 0-100 km/h acceleration: normal range is 3-15 seconds
- If you see obviously wrong values (like 0.8 seconds), mark as suspicious
- Always double-check technical specs against automotive standards

Use ONLY the following document content to answer questions. If documents don't contain the answer, say you don't know and suggest what additional information might be needed.

Document Content:
{{context}}

Question:
{{question}}

{structured_instruction}

IMPORTANT: Respond in Chinese with sentence-level citations 【来源：DOC_X】 for every factual statement.""",

            "features": f"""You are a professional automotive product strategy expert with strict accuracy requirements.

CRITICAL RULES:
1. Only use information explicitly mentioned in the provided documents
2. Analysis must be based on evidence found in documents
3. Never make assumptions beyond document content
4. If documents lack relevant information, clearly state this limitation

{citation_requirement}

Your task is to analyze whether a feature should be added, strictly based on provided document content.

Please analyze feature requirements in two sections:
【Evidence Analysis】 - Evidence-based analysis from provided documents (with citations)
【Strategic Reasoning】 - Strategic reasoning based on found evidence (with citations)

Be factual and cite specific sources. Do not make assumptions beyond document content.

Document Content:
{{context}}

Feature Question:
{{question}}

{structured_instruction}

IMPORTANT: Respond in Chinese with sentence-level citations 【来源：DOC_X】 for evidence.""",

            "tradeoffs": f"""You are a professional automotive design decision analyst with strict accuracy requirements.

CRITICAL RULES:
1. Only use information explicitly mentioned in the provided documents
2. Pros/cons analysis must be based on document evidence
3. Never speculate beyond document content
4. If documents lack sufficient comparison information, clearly state this

{citation_requirement}

Your task is to analyze design choice pros and cons, strictly based on provided document content.

Please analyze in two sections:
【Document Evidence】 - Evidence from provided documents (with citations)
【Pros/Cons Analysis】 - Pros and cons analysis based on evidence (with citations)

Be objective and cite specific sources. Do not speculate beyond document content.

Document Content:
{{context}}

Design Decision Question:
{{question}}

{structured_instruction}

IMPORTANT: Respond in Chinese with sentence-level citations 【来源：DOC_X】 for all evidence.""",

            "scenarios": f"""You are a professional automotive user experience analyst with strict accuracy requirements.

CRITICAL RULES:
1. Only use information explicitly mentioned in the provided documents
2. Scenario analysis must be based on document evidence
3. Never create scenarios not mentioned in documents
4. If documents lack relevant scenario information, clearly state this

{citation_requirement}

Your task is to analyze feature performance in real usage scenarios, strictly based on provided document content.

Please analyze in two sections:
【Document Scenarios】 - Scenarios mentioned in provided documents (with citations)
【Scenario Reasoning】 - Scenario analysis based on found evidence (with citations)

Be specific and cite sources. Do not create scenarios not mentioned in documents.

Document Content:
{{context}}

Scenario Question:
{{question}}

{structured_instruction}

IMPORTANT: Respond in Chinese with sentence-level citations 【来源：DOC_X】 for all scenarios.""",

            "debate": f"""You are a professional automotive industry roundtable discussion moderator with strict accuracy requirements.

CRITICAL RULES:
1. Only use information explicitly mentioned in the provided documents
2. Viewpoints must be based on evidence found in documents
3. Never fabricate viewpoints not supported by documents
4. If documents lack sufficient multi-perspective analysis information, clearly state this

{citation_requirement}

Your task is to present different professional perspectives based on provided document content.

Please present viewpoints from these perspectives:
**👔 Product Manager Perspective:** Based on evidence in documents (with citations)
**🔧 Engineer Perspective:** Based on technical information in documents (with citations)
**👥 User Representative Perspective:** Based on user feedback in documents (with citations)

**📋 Discussion Summary:** Only synthesize content that documents can support (with citations)

Be factual and cite specific sources for each perspective.

Document Content:
{{context}}

Discussion Topic:
{{question}}

{structured_instruction}

IMPORTANT: Respond in Chinese with sentence-level citations 【来源：DOC_X】 for all viewpoints.""",

            "quotes": f"""You are a professional automotive market research analyst with strict accuracy requirements.

CRITICAL RULES:
1. Only extract quotes that actually exist in the provided documents
2. Use exact quotations - do not rewrite or modify
3. Never create or fabricate quotes
4. If no relevant quotes found, clearly state this

{citation_requirement}

Your task is to extract actual user quotes and feedback from provided document content.

Please extract quotes in this format:
【来源：DOC_1】："Exact quote from documents..."
【来源：DOC_2】："Another exact quote from documents..."

If no relevant user quotes found, state: "According to provided documents, no relevant user comments or feedback found."

CRITICAL: Only extract quotes that actually exist in documents. Do not create or rewrite content.

Document Content:
{{context}}

Quote Topic:
{{question}}

{structured_instruction}

IMPORTANT: Respond in Chinese with exact document citations 【来源：DOC_X】 for all quotes."""
        }

        return templates.get(mode, templates["facts"])

    def answer_query_with_mode(
            self,
            query: str,
            documents: List[Tuple[Document, float]],
            query_mode: str = "facts",
            structured_output: bool = False,
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        ENHANCED: Answer a query with advanced anti-hallucination features.

        Args:
            query: The user's query
            documents: Retrieved documents with scores
            query_mode: The query mode to use (defaults to "facts")
            structured_output: Whether to return structured JSON output
            metadata_filter: Optional metadata filters

        Returns:
            Generated answer (string) or structured output (dict) with enhanced fact checking
        """
        # Validate mode (fallback to facts)
        if not self.validate_mode(query_mode):
            logger.warning(f"Invalid query mode '{query_mode}', using facts mode")
            query_mode = "facts"

        # Use enhanced anti-hallucination approach
        answer = self._answer_with_enhanced_anti_hallucination(query, documents, query_mode, structured_output,
                                                               metadata_filter)

        if structured_output:
            # Generate structured JSON output
            structured_data = self.structured_generator.extract_structured_data(answer, documents)
            return {
                "natural_language_answer": answer,
                "structured_data": structured_data,
                "query_mode": query_mode,
                "has_contradictions": len(self.fact_checker.contradiction_detector.detect_contradictions(documents)) > 0
            }
        else:
            return answer

    def answer_query_with_mode_specific_params(
            self,
            query: str,
            documents: List[Tuple[Document, float]],
            query_mode: str = "facts",
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            metadata_filter: Optional[Dict] = None,
    ) -> str:
        """
        Enhanced answer generation with mode-specific parameters.

        This method allows dynamic parameter adjustment per query mode.
        """
        # Parse mode
        try:
            mode_enum = QueryMode(query_mode)
        except ValueError:
            mode_enum = QueryMode.FACTS

        # Get mode-specific defaults if not provided
        if None in [temperature, max_tokens, top_p, repetition_penalty]:
            mode_defaults = mode_config.get_llm_params(mode_enum)
            temperature = temperature or mode_defaults["temperature"]
            max_tokens = max_tokens or mode_defaults["max_tokens"]
            top_p = top_p or mode_defaults.get("top_p", 0.85)
            repetition_penalty = repetition_penalty or mode_defaults.get("repetition_penalty", 1.1)

        logger.info(f"LLM inference with mode-specific params: T={temperature}, max_tokens={max_tokens}, top_p={top_p}")

        # Check for contradictions
        contradictions = self.fact_checker.contradiction_detector.detect_contradictions(documents)

        # Get mode-specific template
        template = self.get_prompt_template_for_mode(query_mode, structured_output=False)

        # Format context with relevance scores and token budget
        context = get_context_with_token_budget(documents, query_mode)

        # Add contradiction warning if needed
        if contradictions:
            contradiction_warning = "\n\nIMPORTANT: Contradictions detected in documents:\n"
            for spec_name, contradiction_info in contradictions.items():
                spec_chinese = contradiction_info["spec_name_chinese"]
                values_info = ", ".join([
                    f"{item['value']} ({item['doc_id']})"
                    for item in contradiction_info["conflicting_values"]
                ])
                contradiction_warning += f"- {spec_chinese}: {values_info}\n"
            contradiction_warning += "Please acknowledge these contradictions in your response.\n"
            context += contradiction_warning

        # Create prompt
        prompt = template.format(context=context, question=query)

        # Generate with mode-specific parameters
        start_time = time.time()

        try:
            # Use custom pipeline parameters
            results = self.pipe(
                prompt,
                num_return_sequences=1,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_tokens
            )

            answer = results[0]["generated_text"]

            # Clean answer
            if answer.startswith("</think>\n\n"):
                answer = answer.replace("</think>\n\n", "").strip()
            if answer.startswith("<think>") and "</think>" in answer:
                answer = answer.split("</think>")[-1].strip()

            # Enhanced fact checking with relevance consideration
            quality_check = self.fact_checker.check_answer_quality(answer, context, documents)

            # Add mode-specific quality adjustments
            if mode_enum == QueryMode.FACTS and quality_check["quality_score"] < 70:
                # More stringent quality requirements for facts mode
                logger.warning(f"Facts mode quality below threshold: {quality_check['quality_score']}")

            # Add quality disclaimer if needed
            if quality_check["has_issues"]:
                disclaimer = "\n\n⚠️ 注意: 此答案可能包含需要验证的信息，建议查阅更多资料确认。"
                answer += disclaimer

            # Add contradiction summary if detected
            if contradictions:
                contradiction_summary = "\n\n📋 数据冲突提醒:\n"
                for spec_name, contradiction_info in contradictions.items():
                    spec_chinese = contradiction_info["spec_name_chinese"]
                    recommendation = contradiction_info["recommendation"]
                    contradiction_summary += f"• {spec_chinese}: {recommendation}\n"
                answer += contradiction_summary

            generation_time = time.time() - start_time
            logger.info(
                f"Mode '{query_mode}' generation completed in {generation_time:.2f}s (Quality: {quality_check['quality_score']:.1f})")

            return answer

        except Exception as e:
            logger.error(f"Generation failed for mode '{query_mode}': {e}")
            raise e

    def calculate_enhanced_confidence(
            self,
            answer: str,
            documents: List[Tuple[Document, float]],
            query_mode: str,
            avg_relevance_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        Enhanced confidence calculation that incorporates relevance scores.

        This creates the bridge between document relevance and answer confidence.
        """
        # Get context with proper formatting
        context = get_context_with_token_budget(documents, query_mode)

        # Get basic confidence metrics
        base_confidence = self.confidence_scorer.calculate_confidence(answer, context, documents)

        # Enhanced metrics with relevance correlation
        enhanced_metrics = base_confidence.copy()

        # Add relevance-confidence correlation
        relevance_confidence_factor = min(1.0, avg_relevance_score * 2)  # Scale 0.5->1.0 relevance to 1.0 factor

        # Adjust confidence based on relevance
        adjusted_confidence = base_confidence["overall_confidence"] * (0.7 + 0.3 * relevance_confidence_factor)
        adjusted_confidence = min(100, adjusted_confidence)

        # Mode-specific confidence adjustments
        try:
            mode_enum = QueryMode(query_mode)
            mode_complexity = mode_config.get_mode_complexity(mode_enum)

            if mode_complexity == "complex" and adjusted_confidence > 90:
                # Slightly reduce confidence for complex modes (harder to be certain)
                adjusted_confidence *= 0.95
            elif mode_complexity == "simple" and avg_relevance_score > 0.7:
                # Boost confidence for simple modes with high relevance
                adjusted_confidence = min(100, adjusted_confidence * 1.05)
        except:
            pass

        enhanced_metrics.update({
            "relevance_adjusted_confidence": adjusted_confidence,
            "avg_document_relevance": avg_relevance_score,
            "relevance_confidence_correlation": abs(avg_relevance_score - adjusted_confidence / 100),
            "mode_complexity_factor": mode_config.get_mode_complexity(
                QueryMode(query_mode)) if query_mode else "unknown"
        })

        return enhanced_metrics

    def _answer_with_enhanced_anti_hallucination(
            self,
            query: str,
            documents: List[Tuple[Document, float]],
            query_mode: str,
            structured_output: bool = False,
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
    ) -> str:
        """
        ENHANCED: Answer generation with comprehensive anti-hallucination measures including contradiction detection.
        """
        # ENHANCED: Check for contradictions in documents first
        contradictions = self.fact_checker.contradiction_detector.detect_contradictions(documents)

        # Get the appropriate template for this mode
        template = self.get_prompt_template_for_mode(query_mode, structured_output)

        # Format documents into context with document IDs
        context = get_context_with_token_budget(documents, query_mode)

        # Add contradiction warning to prompt if needed
        if contradictions:
            contradiction_warning = "\n\nIMPORTANT: The following contradictions were detected in the documents:\n"
            for spec_name, contradiction_info in contradictions.items():
                spec_chinese = contradiction_info["spec_name_chinese"]
                values_info = ", ".join([
                    f"{item['value']} ({item['doc_id']})"
                    for item in contradiction_info["conflicting_values"]
                ])
                contradiction_warning += f"- {spec_chinese}: {values_info}\n"
            contradiction_warning += "Please acknowledge these contradictions in your response.\n"

            context += contradiction_warning

        # Create prompt using the mode-specific template
        prompt = template.format(
            context=context,
            question=query
        )

        # Generate initial answer with lower temperature for more deterministic output
        start_time = time.time()

        try:
            # First attempt with conservative settings
            results = self.pipe(
                prompt,
                num_return_sequences=1,
                do_sample=True,
                temperature=max(0.0, self.temperature - 0.05),  # Even lower temperature
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_tokens
            )

            initial_answer = results[0]["generated_text"]

            # Clean the answer
            if initial_answer.startswith("</think>\n\n"):
                initial_answer = initial_answer.replace("</think>\n\n", "").strip()
            if initial_answer.startswith("<think>") and "</think>" in initial_answer:
                initial_answer = initial_answer.split("</think>")[-1].strip()

            # ENHANCED: Perform fact checking with contradiction detection
            quality_check = self.fact_checker.check_answer_quality(initial_answer, context, documents)

            # If serious issues detected, regenerate with stricter prompt
            if quality_check["has_issues"] and quality_check["quality_score"] < 70:
                logger.warning(f"Fact checking detected issues for mode '{query_mode}': {quality_check['warnings']}")

                # Use stricter prompt for regeneration
                strict_prompt = self._create_enhanced_strict_verification_prompt(query, context,
                                                                                 quality_check["warnings"],
                                                                                 contradictions)

                try:
                    strict_results = self.pipe(
                        strict_prompt,
                        num_return_sequences=1,
                        do_sample=False,  # Use greedy decoding for maximum determinism
                        temperature=0.0,  # Zero temperature
                        pad_token_id=self.tokenizer.eos_token_id,
                        max_new_tokens=self.max_tokens
                    )

                    regenerated_answer = strict_results[0]["generated_text"]

                    # Re-check the regenerated answer
                    second_check = self.fact_checker.check_answer_quality(regenerated_answer, context, documents)

                    if second_check["quality_score"] > quality_check["quality_score"]:
                        logger.info(
                            f"Regenerated answer improved quality score from {quality_check['quality_score']:.1f} to {second_check['quality_score']:.1f}")
                        final_answer = regenerated_answer
                        final_quality = second_check
                    else:
                        logger.warning("Regeneration did not improve quality, using original")
                        final_answer = initial_answer
                        final_quality = quality_check

                except Exception as e:
                    logger.error(f"Answer regeneration failed for mode '{query_mode}': {e}")
                    final_answer = initial_answer
                    final_quality = quality_check
            else:
                final_answer = initial_answer
                final_quality = quality_check

            # Add quality disclaimer if issues remain
            if final_quality["has_issues"]:
                disclaimer = "\n\n⚠️ 注意: 此答案可能包含需要验证的信息，建议查阅更多资料确认。"
                final_answer += disclaimer

            # ENHANCED: Add contradiction summary if detected
            if contradictions:
                contradiction_summary = "\n\n📋 数据冲突提醒:\n"
                for spec_name, contradiction_info in contradictions.items():
                    spec_chinese = contradiction_info["spec_name_chinese"]
                    recommendation = contradiction_info["recommendation"]
                    contradiction_summary += f"• {spec_chinese}: {recommendation}\n"
                final_answer += contradiction_summary

            generation_time = time.time() - start_time
            print(
                f"Enhanced mode '{query_mode}' answer generated in {generation_time:.2f} seconds (Quality Score: {final_quality['quality_score']:.1f})")

            return final_answer

        except Exception as e:
            print(f"❌ Generation failed for mode '{query_mode}': {e}")
            if "CUDA" in str(e):
                print("This may be a Tesla T4 memory or quantization issue.")
                print("Check your environment settings:")
                print(f"  LLM_USE_4BIT: {settings.llm_use_4bit}")
                print(f"  GPU_MEMORY_FRACTION_INFERENCE: {settings.gpu_memory_fraction_inference}")
            raise e

    def _create_enhanced_strict_verification_prompt(self, query: str, context: str, warnings: List[str],
                                                    contradictions: Dict) -> str:
        """
        ENHANCED: Create strict English prompt for answer regeneration with contradiction awareness.
        """
        warnings_text = "\n".join(f"- {warning}" for warning in warnings)

        contradiction_text = ""
        if contradictions:
            contradiction_text = "\n\nDETECTED CONTRADICTIONS:\n"
            for spec_name, contradiction_info in contradictions.items():
                spec_chinese = contradiction_info["spec_name_chinese"]
                values_info = ", ".join([
                    f"{item['value']} (from {item['doc_id']})"
                    for item in contradiction_info["conflicting_values"]
                ])
                contradiction_text += f"- {spec_chinese}: {values_info}\n"

        prompt = f"""As an automotive specifications expert, please answer the question based on the following documents.

CRITICAL: Previous answer detected these issues:
{warnings_text}

{contradiction_text}

ENHANCED STRICT REQUIREMENTS:
1. Only use information explicitly mentioned in documents
2. If documents lack specific data, clearly state "Documents do not mention this data"
3. Do not guess or infer any numerical values
4. If you find unreasonable data, question its accuracy
5. All numerical values must be traceable to document content
6. MANDATORY: Use sentence-level citations 【来源：DOC_X】 for every factual statement
7. If contradictions exist, acknowledge them and present both values with sources

Document Content:
{context}

Question: {query}

IMPORTANT: Provide accurate, evidence-based Chinese response with sentence-level citations 【来源：DOC_X】 for every fact:"""

        return prompt

    def answer_with_confidence_scoring(
            self,
            query: str,
            documents: List[Tuple[Document, float]],
            query_mode: str = "facts",
            structured_output: bool = False,
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
    ) -> Dict[str, any]:
        """
        ENHANCED: Generate answer with confidence scoring, contradiction detection, and optional structured output.
        """
        # Generate the answer with enhanced anti-hallucination measures
        answer = self._answer_with_enhanced_anti_hallucination(query, documents, query_mode, structured_output,
                                                               metadata_filter)

        # Calculate confidence using quality_utils functions and enhanced fact checking
        context = get_context_with_token_budget(documents, query_mode)
        confidence_metrics = self.confidence_scorer.calculate_confidence(answer, context, documents)

        # Detect contradictions
        contradictions = self.fact_checker.contradiction_detector.detect_contradictions(documents)

        # Generate structured output if requested
        structured_data = None
        if structured_output:
            structured_data = self.structured_generator.extract_structured_data(answer, documents)

        # Prepare enhanced response
        response = {
            'answer': answer,
            'confidence_metrics': confidence_metrics,
            'contradictions': contradictions,
            'query_mode': query_mode,
            'document_count': len(documents),
            'timestamp': time.time(),
            'enhanced_features': {
                'sentence_level_citations': True,
                'contradiction_detection': len(contradictions) > 0,
                'structured_output_available': structured_output
            }
        }

        if structured_data:
            response['structured_data'] = structured_data

        # Add warning if confidence is low or contradictions exist
        warnings = []
        if confidence_metrics['should_flag']:
            warnings.append(
                f"⚠️ 置信度较低 ({confidence_metrics['overall_confidence']:.1f}%), {confidence_metrics['recommendation']}")

        if contradictions:
            warnings.append(f"⚠️ 检测到 {len(contradictions)} 项数据冲突，请注意核实")

        if warnings:
            warning_text = "\n\n" + "\n".join(warnings)
            response['answer'] = f"{answer}{warning_text}"
            response['flagged_for_review'] = True
        else:
            response['flagged_for_review'] = False

        return response

    def validate_mode(self, mode: str) -> bool:
        """Validate if the query mode is supported."""
        valid_modes = ["facts", "features", "tradeoffs", "scenarios", "debate", "quotes"]
        return mode in valid_modes

    def get_mode_info(self, mode: str) -> Dict[str, Any]:
        """Get information about a specific query mode."""
        mode_info = {
            "facts": {
                "name": "车辆规格查询",
                "description": "直接验证具体的车辆规格参数",
                "two_layer": False,
                "complexity": "simple",
                "template_type": "enhanced_english_anti_hallucination_chinese_response",
                "is_default": True,
                "anti_hallucination": True,
                "language": "chinese_response_english_template",
                "enhanced_features": ["sentence_level_citations", "contradiction_detection", "structured_output"]
            },
            "features": {
                "name": "新功能建议",
                "description": "评估是否应该添加某项功能",
                "two_layer": True,
                "complexity": "moderate",
                "template_type": "enhanced_english_structured_analysis_chinese_response",
                "is_default": False,
                "anti_hallucination": True,
                "language": "chinese_response_english_template",
                "enhanced_features": ["sentence_level_citations", "contradiction_detection", "structured_output"]
            },
            "tradeoffs": {
                "name": "权衡利弊分析",
                "description": "分析设计选择的优缺点",
                "two_layer": True,
                "complexity": "complex",
                "template_type": "enhanced_english_structured_analysis_chinese_response",
                "is_default": False,
                "anti_hallucination": True,
                "language": "chinese_response_english_template",
                "enhanced_features": ["sentence_level_citations", "contradiction_detection", "structured_output"]
            },
            "scenarios": {
                "name": "用户场景分析",
                "description": "评估功能在实际使用场景中的表现",
                "two_layer": True,
                "complexity": "complex",
                "template_type": "enhanced_english_structured_analysis_chinese_response",
                "is_default": False,
                "anti_hallucination": True,
                "language": "chinese_response_english_template",
                "enhanced_features": ["sentence_level_citations", "contradiction_detection", "structured_output"]
            },
            "debate": {
                "name": "多角色讨论",
                "description": "模拟不同角色的观点和讨论",
                "two_layer": False,
                "complexity": "complex",
                "template_type": "enhanced_english_multi_perspective_chinese_response",
                "is_default": False,
                "anti_hallucination": True,
                "language": "chinese_response_english_template",
                "enhanced_features": ["sentence_level_citations", "contradiction_detection", "structured_output"]
            },
            "quotes": {
                "name": "原始用户评论",
                "description": "提取相关的用户评论和反馈",
                "two_layer": False,
                "complexity": "simple",
                "template_type": "enhanced_english_extraction_chinese_response",
                "is_default": False,
                "anti_hallucination": True,
                "language": "chinese_response_english_template",
                "enhanced_features": ["sentence_level_citations", "contradiction_detection", "structured_output"]
            }
        }

        return mode_info.get(mode, mode_info["facts"])

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model including enhanced features."""
        memory_info = {}

        # Get GPU memory usage if available
        if self.device.startswith("cuda") and torch.cuda.is_available():
            device_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
            memory_allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)
            total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)

            memory_info.update({
                "memory_allocated_gb": f"{memory_allocated:.2f}",
                "memory_reserved_gb": f"{memory_reserved:.2f}",
                "total_memory_gb": f"{total_memory:.2f}",
                "memory_utilization": f"{(memory_allocated / total_memory) * 100:.1f}%"
            })

        # Model configuration info including enhanced features
        model_config = {
            "model_name": self.model_name,
            "device": self.device,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "quantization": "4-bit" if self.use_4bit else "8-bit" if self.use_8bit else "none",
            "torch_dtype": str(self.torch_dtype),
            "use_fp16": settings.use_fp16,
            "environment_driven": True,
            "worker_type": os.environ.get("WORKER_TYPE", "unknown"),
            "memory_fraction": settings.get_worker_memory_fraction(),
            "tesla_t4_optimized": not self.use_4bit,
            "query_system": "unified_enhanced_ADVANCED_anti_hallucination",
            "default_mode": "facts",
            "template_system": "ENHANCED_english_anti_hallucination_chinese_output",
            "response_language": "chinese",
            "template_language": "english",
            "token_efficiency": "optimized_english_templates",
            "code_architecture": "DEDUPED_quality_utils_integration",
            "supported_modes": ["facts", "features", "tradeoffs", "scenarios", "debate", "quotes"],
            "enhanced_anti_hallucination_features": {
                "fact_checker": True,
                "confidence_scorer": True,
                "strict_prompts": True,
                "context_verification": True,
                "numerical_validation": True,
                "regeneration_on_issues": True,
                "english_templates": True,
                "chinese_responses": True,
                "automotive_domain_knowledge": True,
                "token_optimized": True,
                "code_deduplication": "COMPLETE",
                "shared_utilities": "quality_utils.py",
                # ENHANCED FEATURES
                "sentence_level_citations": True,
                "contradiction_detection": True,
                "structured_json_output": True,
                "advanced_citation_tracking": True,
                "multi_source_validation": True,
                "enhanced_quality_scoring": True,
                "mode_specific_parameters": True,
                "relevance_score_integration": True,
                "token_budget_management": True
            }
        }

        return {**model_config, **memory_info}