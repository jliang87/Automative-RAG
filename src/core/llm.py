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
        documents: List[Tuple[Document, float]]
) -> str:
    """
    Format retrieved documents into context for the prompt.
    """
    context_parts = []

    for i, (doc, score) in enumerate(documents):
        # Extract metadata for citation
        metadata = doc.metadata
        source_type = metadata.get("source", "unknown")
        title = metadata.get("title", f"Document {i + 1}")

        # Format source information
        if source_type == "youtube":
            source_info = f"Source {i + 1}: YouTube - '{title}'"
            if "url" in metadata:
                source_info += f" ({metadata['url']})"
        elif source_type == "bilibili":
            source_info = f"Source {i + 1}: Bilibili - '{title}'"
            if "url" in metadata:
                source_info += f" ({metadata['url']})"
        elif source_type == "pdf":
            source_info = f"Source {i + 1}: PDF - '{title}'"
        else:
            source_info = f"Source {i + 1}: {title}"

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

        # Format content block
        content_block = f"{source_info}\n{doc.page_content}\n"
        context_parts.append(content_block)

    return "\n\n".join(context_parts)


class AutomotiveFactChecker:
    """
    Fact checker for automotive specifications to detect obvious hallucinations.
    DEDUPED: Uses shared utility functions from quality_utils.py
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

    def check_answer_quality(self, answer: str, context: str) -> Dict[str, any]:
        """
        Comprehensive answer quality check using shared utility functions.
        DEDUPED: All actual checking is done by quality_utils functions.
        """
        warnings = []

        # Use shared utility functions - NO DUPLICATION
        warnings.extend(check_acceleration_claims(answer))
        warnings.extend(check_numerical_specs_realistic(answer))
        warnings.extend(self._verify_context_support(answer, context))

        # Calculate quality score
        quality_score = max(0, 100 - len(warnings) * 15)

        return {
            "warnings": warnings,
            "quality_score": quality_score,
            "has_issues": len(warnings) > 0,
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
        DEDUPED: Uses quality_utils functions where possible.
        """
        scores = {}

        # 1. Context Support Score (0-100) - uses quality_utils
        scores['context_support'] = self._calculate_context_support(answer, context)

        # 2. Document Relevance Score (0-100)
        scores['document_relevance'] = self._calculate_document_relevance(answer, documents)

        # 3. Factual Consistency Score (0-100) - uses quality_utils via fact_checker
        scores['factual_consistency'] = self._calculate_factual_consistency(answer, context)

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

    def _calculate_factual_consistency(self, answer: str, context: str) -> float:
        """Check for factual consistency using the fact checker."""
        quality_check = self.fact_checker.check_answer_quality(answer, context)

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

    DEDUPED: English templates with explicit Chinese response instruction.
    Uses quality_utils.py for all automotive domain logic.
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

        # Initialize fact checker and confidence scorer
        self.fact_checker = AutomotiveFactChecker()
        self.confidence_scorer = AnswerConfidenceScorer()

        # Log configuration for debugging
        print(f"LocalLLM Configuration (DEDUPED: Uses quality_utils.py):")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Use 4-bit: {self.use_4bit}")
        print(f"  Use 8-bit: {self.use_8bit}")
        print(f"  Torch dtype: {self.torch_dtype}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Template Language: ENGLISH (token efficient)")
        print(f"  Response Language: CHINESE (enforced)")
        print(f"  Anti-Hallucination: CHINESE-OPTIMIZED (quality_utils.py)")
        print(f"  Code Duplication: ELIMINATED")

        # Initialize tokenizer and model
        self._load_model()

        # DEDUPED: English templates with Chinese response instruction
        self.qa_prompt_template = self._create_english_anti_hallucination_template()

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

    def _create_english_anti_hallucination_template(self) -> str:
        """
        DEDUPED: Create English template with explicit Chinese response instruction.
        This dramatically reduces token count while ensuring Chinese output.
        """
        template = """You are a professional automotive specifications expert assistant with strict accuracy requirements.

CRITICAL RULES:
1. Only use information explicitly mentioned in the provided documents
2. If specific numbers/specs are not in documents, say "According to provided documents, specific [parameter] data not found"
3. Never estimate, guess, or infer any numerical values
4. If document content is unclear or contradictory, acknowledge this uncertainty
5. Always cite the exact sources where information was found

NUMERICAL ACCURACY CHECK:
- 0-100 km/h acceleration: normal range is 3-15 seconds
- If you see obviously wrong values (like 0.8 seconds), mark as suspicious
- Always double-check technical specs against automotive standards

Your task is to help users find automotive specifications, features, and technical details.

Use ONLY the following document content to answer questions. If documents don't contain the answer, say you don't know and suggest what additional information might be needed.

Document Content:
{context}

Question:
{question}

IMPORTANT: You must respond in Chinese, but be precise and factual. Cite specific sources (document titles or URLs) where you found the information.

Response Format:
1. Direct answer based on documents (or "information not found")
2. Source citations
3. If uncertain, clearly state limitations"""
        return template

    def get_prompt_template_for_mode(self, mode: str) -> str:
        """
        DEDUPED: Get English prompt templates with Chinese response instruction.
        Much more token-efficient than Chinese templates.
        """

        templates = {
            "facts": """You are a professional automotive specifications expert assistant with strict accuracy requirements.

CRITICAL RULES:
1. Only use information explicitly mentioned in the provided documents
2. If specific numbers/specs are not in documents, say "According to provided documents, specific [parameter] data not found"
3. Never estimate, guess, or infer any numerical values
4. If document content is unclear or contradictory, acknowledge this uncertainty
5. Always cite the exact sources where information was found

NUMERICAL ACCURACY CHECK:
- 0-100 km/h acceleration: normal range is 3-15 seconds
- If you see obviously wrong values (like 0.8 seconds), mark as suspicious
- Always double-check technical specs against automotive standards

Use ONLY the following document content to answer questions. If documents don't contain the answer, say you don't know and suggest what additional information might be needed.

Document Content:
{context}

Question:
{question}

IMPORTANT: Respond in Chinese and cite specific sources (document titles or URLs).""",

            "features": """You are a professional automotive product strategy expert with strict accuracy requirements.

CRITICAL RULES:
1. Only use information explicitly mentioned in the provided documents
2. Analysis must be based on evidence found in documents
3. Never make assumptions beyond document content
4. If documents lack relevant information, clearly state this limitation

Your task is to analyze whether a feature should be added, strictly based on provided document content.

Please analyze feature requirements in two sections:
【Evidence Analysis】 - Evidence-based analysis from provided documents
【Strategic Reasoning】 - Strategic reasoning based on found evidence

Be factual and cite specific sources. Do not make assumptions beyond document content.

Document Content:
{context}

Feature Question:
{question}

IMPORTANT: Respond in Chinese and cite specific sources (document titles or URLs).""",

            "tradeoffs": """You are a professional automotive design decision analyst with strict accuracy requirements.

CRITICAL RULES:
1. Only use information explicitly mentioned in the provided documents
2. Pros/cons analysis must be based on document evidence
3. Never speculate beyond document content
4. If documents lack sufficient comparison information, clearly state this

Your task is to analyze design choice pros and cons, strictly based on provided document content.

Please analyze in two sections:
【Document Evidence】 - Evidence from provided documents
【Pros/Cons Analysis】 - Pros and cons analysis based on evidence

Be objective and cite specific sources. Do not speculate beyond document content.

Document Content:
{context}

Design Decision Question:
{question}

IMPORTANT: Respond in Chinese and cite specific sources (document titles or URLs).""",

            "scenarios": """You are a professional automotive user experience analyst with strict accuracy requirements.

CRITICAL RULES:
1. Only use information explicitly mentioned in the provided documents
2. Scenario analysis must be based on document evidence
3. Never create scenarios not mentioned in documents
4. If documents lack relevant scenario information, clearly state this

Your task is to analyze feature performance in real usage scenarios, strictly based on provided document content.

Please analyze in two sections:
【Document Scenarios】 - Scenarios mentioned in provided documents
【Scenario Reasoning】 - Scenario analysis based on found evidence

Be specific and cite sources. Do not create scenarios not mentioned in documents.

Document Content:
{context}

Scenario Question:
{question}

IMPORTANT: Respond in Chinese and cite specific sources (document titles or URLs).""",

            "debate": """You are a professional automotive industry roundtable discussion moderator with strict accuracy requirements.

CRITICAL RULES:
1. Only use information explicitly mentioned in the provided documents
2. Viewpoints must be based on evidence found in documents
3. Never fabricate viewpoints not supported by documents
4. If documents lack sufficient multi-perspective analysis information, clearly state this

Your task is to present different professional perspectives based on provided document content.

Please present viewpoints from these perspectives:
**👔 Product Manager Perspective:** Based on evidence in documents
**🔧 Engineer Perspective:** Based on technical information in documents
**👥 User Representative Perspective:** Based on user feedback in documents

**📋 Discussion Summary:** Only synthesize content that documents can support

Be factual and cite specific sources for each perspective.

Document Content:
{context}

Discussion Topic:
{question}

IMPORTANT: Respond in Chinese and cite specific sources (document titles or URLs).""",

            "quotes": """You are a professional automotive market research analyst with strict accuracy requirements.

CRITICAL RULES:
1. Only extract quotes that actually exist in the provided documents
2. Use exact quotations - do not rewrite or modify
3. Never create or fabricate quotes
4. If no relevant quotes found, clearly state this

Your task is to extract actual user quotes and feedback from provided document content.

Please extract quotes in this format:
【Source 1】："Exact quote from documents..."
【Source 2】："Another exact quote from documents..."

If no relevant user quotes found, state: "According to provided documents, no relevant user comments or feedback found."

CRITICAL: Only extract quotes that actually exist in documents. Do not create or rewrite content.

Document Content:
{context}

Quote Topic:
{question}

IMPORTANT: Respond in Chinese and cite specific sources (document titles or URLs)."""
        }

        return templates.get(mode, templates["facts"])

    def answer_query_with_mode(
            self,
            query: str,
            documents: List[Tuple[Document, float]],
            query_mode: str = "facts",
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
    ) -> str:
        """
        UNIFIED: Answer a query using a specific mode template with anti-hallucination features.
        DEDUPED: All automotive logic delegated to quality_utils.py
        """
        # Validate mode (fallback to facts)
        if not self.validate_mode(query_mode):
            logger.warning(f"Invalid query mode '{query_mode}', using facts mode")
            query_mode = "facts"

        # Use enhanced anti-hallucination approach
        return self._answer_with_anti_hallucination(query, documents, query_mode, metadata_filter)

    def _answer_with_anti_hallucination(
            self,
            query: str,
            documents: List[Tuple[Document, float]],
            query_mode: str,
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
    ) -> str:
        """
        Enhanced answer generation with comprehensive anti-hallucination measures.
        DEDUPED: Uses quality_utils.py functions for all domain-specific checks.
        """
        # Get the appropriate template for this mode
        template = self.get_prompt_template_for_mode(query_mode)

        # Format documents into context
        context = _format_documents_for_context(documents)

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

            # DEDUPED: Perform fact checking using quality_utils functions
            quality_check = self.fact_checker.check_answer_quality(initial_answer, context)

            # If serious issues detected, regenerate with stricter prompt
            if quality_check["has_issues"] and quality_check["quality_score"] < 70:
                logger.warning(f"Fact checking detected issues for mode '{query_mode}': {quality_check['warnings']}")

                # Use stricter prompt for regeneration
                strict_prompt = self._create_strict_verification_prompt(query, context, quality_check["warnings"])

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
                    second_check = self.fact_checker.check_answer_quality(regenerated_answer, context)

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

            generation_time = time.time() - start_time
            print(
                f"Mode '{query_mode}' answer generated in {generation_time:.2f} seconds (Quality Score: {final_quality['quality_score']:.1f})")

            return final_answer

        except Exception as e:
            print(f"❌ Generation failed for mode '{query_mode}': {e}")
            if "CUDA" in str(e):
                print("This may be a Tesla T4 memory or quantization issue.")
                print("Check your environment settings:")
                print(f"  LLM_USE_4BIT: {settings.llm_use_4bit}")
                print(f"  GPU_MEMORY_FRACTION_INFERENCE: {settings.gpu_memory_fraction_inference}")
            raise e

    def _create_strict_verification_prompt(self, query: str, context: str, warnings: List[str]) -> str:
        """
        DEDUPED: Create strict English prompt for answer regeneration.
        """
        warnings_text = "\n".join(f"- {warning}" for warning in warnings)

        prompt = f"""As an automotive specifications expert, please answer the question based on the following documents.

CRITICAL: Previous answer detected these issues:
{warnings_text}

STRICT REQUIREMENTS:
1. Only use information explicitly mentioned in documents
2. If documents lack specific data, clearly state "Documents do not mention this data"
3. Do not guess or infer any numerical values
4. If you find unreasonable data, question its accuracy
5. All numerical values must be traceable to document content

Document Content:
{context}

Question: {query}

IMPORTANT: Provide accurate, evidence-based Chinese response with specific source citations:"""

        return prompt

    def answer_with_confidence_scoring(
            self,
            query: str,
            documents: List[Tuple[Document, float]],
            query_mode: str = "facts",
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
    ) -> Dict[str, any]:
        """
        Generate answer with confidence scoring and quality assessment.
        DEDUPED: Uses quality_utils.py functions via confidence scorer.
        """
        # Generate the answer with anti-hallucination measures
        answer = self._answer_with_anti_hallucination(query, documents, query_mode, metadata_filter)

        # Calculate confidence using quality_utils functions
        context = _format_documents_for_context(documents)
        confidence_metrics = self.confidence_scorer.calculate_confidence(answer, context, documents)

        # Prepare enhanced response
        response = {
            'answer': answer,
            'confidence_metrics': confidence_metrics,
            'query_mode': query_mode,
            'document_count': len(documents),
            'timestamp': time.time()
        }

        # Add warning if confidence is low
        if confidence_metrics['should_flag']:
            warning = f"⚠️ 置信度较低 ({confidence_metrics['overall_confidence']:.1f}%), {confidence_metrics['recommendation']}"
            response['answer'] = f"{answer}\n\n{warning}"
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
                "template_type": "english_anti_hallucination_chinese_response",
                "is_default": True,
                "anti_hallucination": True,
                "language": "chinese_response_english_template",
                "deduplication": "quality_utils_integration"
            },
            "features": {
                "name": "新功能建议",
                "description": "评估是否应该添加某项功能",
                "two_layer": True,
                "complexity": "moderate",
                "template_type": "english_structured_analysis_chinese_response",
                "is_default": False,
                "anti_hallucination": True,
                "language": "chinese_response_english_template",
                "deduplication": "quality_utils_integration"
            },
            "tradeoffs": {
                "name": "权衡利弊分析",
                "description": "分析设计选择的优缺点",
                "two_layer": True,
                "complexity": "complex",
                "template_type": "english_structured_analysis_chinese_response",
                "is_default": False,
                "anti_hallucination": True,
                "language": "chinese_response_english_template",
                "deduplication": "quality_utils_integration"
            },
            "scenarios": {
                "name": "用户场景分析",
                "description": "评估功能在实际使用场景中的表现",
                "two_layer": True,
                "complexity": "complex",
                "template_type": "english_structured_analysis_chinese_response",
                "is_default": False,
                "anti_hallucination": True,
                "language": "chinese_response_english_template",
                "deduplication": "quality_utils_integration"
            },
            "debate": {
                "name": "多角色讨论",
                "description": "模拟不同角色的观点和讨论",
                "two_layer": False,
                "complexity": "complex",
                "template_type": "english_multi_perspective_chinese_response",
                "is_default": False,
                "anti_hallucination": True,
                "language": "chinese_response_english_template",
                "deduplication": "quality_utils_integration"
            },
            "quotes": {
                "name": "原始用户评论",
                "description": "提取相关的用户评论和反馈",
                "two_layer": False,
                "complexity": "simple",
                "template_type": "english_extraction_chinese_response",
                "is_default": False,
                "anti_hallucination": True,
                "language": "chinese_response_english_template",
                "deduplication": "quality_utils_integration"
            }
        }

        return mode_info.get(mode, mode_info["facts"])

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model including deduplication status."""
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

        # Model configuration info including deduplication status
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
            "query_system": "unified_enhanced_DEDUPED_english_templates_chinese_responses",
            "default_mode": "facts",
            "template_system": "DEDUPED_english_anti_hallucination_chinese_output",
            "response_language": "chinese",
            "template_language": "english",
            "token_efficiency": "optimized_english_templates",
            "code_architecture": "DEDUPED_quality_utils_integration",
            "supported_modes": ["facts", "features", "tradeoffs", "scenarios", "debate", "quotes"],
            "anti_hallucination_features": {
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
                "shared_utilities": "quality_utils.py"
            }
        }

        return {**model_config, **memory_info}