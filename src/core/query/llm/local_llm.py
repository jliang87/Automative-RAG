import time
import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.config.settings import settings
from src.core.query.llm.mode_config import mode_config, QueryMode, estimate_token_count

logger = logging.getLogger(__name__)


def format_documents_with_relevance_scores(
        documents: List[Tuple[Document, float]],
        max_token_budget: Optional[int] = None
) -> str:
    """
    SIMPLIFIED: Format documents with relevance scores and token management.

    KEY FEATURES:
    1. Shows relevance scores to help LLM assess trustworthiness
    2. Respects token budget limits
    3. Clean, simple formatting

    Args:
        documents: List of (document, relevance_score) tuples
        max_token_budget: Maximum tokens to use for context

    Returns:
        Formatted context string within token limits
    """
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
            source_info = f"{doc_id} (YouTube - '{title}')"
        elif source_type == "bilibili":
            source_info = f"{doc_id} (Bilibili - '{title}')"
        elif source_type == "pdf":
            source_info = f"{doc_id} (PDF - '{title}')"
        else:
            source_info = f"{doc_id} ({title})"

        # Add manufacturer and model if available
        manufacturer = metadata.get("manufacturer")
        model = metadata.get("model")
        if manufacturer or model:
            source_info += " - "
            if manufacturer:
                source_info += manufacturer
            if model:
                source_info += f" {model}"

        # IMPORTANT: Add relevance score to help LLM assess trustworthiness
        confidence_indicator = "🔥" if score > 0.8 else "⭐" if score > 0.6 else "📄"
        source_info += f" {confidence_indicator} (Relevance: {score:.2f})"

        # Create content block
        content_block = f"{source_info}\n{doc.page_content}\n"

        # Token budget management
        if max_token_budget:
            block_tokens = estimate_token_count(content_block)

            # Check if adding this block would exceed budget
            if total_tokens + block_tokens > max_token_budget:
                # Try to include a truncated version if this is high-relevance content
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
        if len(context_parts) >= 12:  # Reasonable limit
            logger.info("Reached maximum document limit (12), stopping context building")
            break

    final_context = "\n\n".join(context_parts)

    # Log context statistics
    final_tokens = estimate_token_count(final_context)
    logger.info(f"Context built: {len(context_parts)} documents, ~{final_tokens} tokens")
    if max_token_budget:
        logger.info(f"Token budget: {final_tokens}/{max_token_budget} ({(final_tokens / max_token_budget) * 100:.1f}%)")

    return final_context


class SimpleFactChecker:
    """
    SIMPLIFIED fact checker focused on obvious hallucination patterns.

    REMOVED: Complex automotive spec ranges, over-engineered validation
    KEPT: Simple content validation, basic reasonableness checks
    """

    def __init__(self):
        # Simple validation patterns for obviously wrong claims
        self.suspicious_patterns = [
            r'(\d+\.?\d*)\s*秒.*?(?:百公里|0-100)',  # Acceleration times
            r'(?:最高时速|极速).*?(\d+)',  # Top speeds
            r'(?:马力|功率).*?(\d+)',  # Horsepower
        ]

    def simple_quality_check(self, answer: str, context: str) -> Dict[str, Any]:
        """
        SIMPLIFIED quality check for obvious issues.

        FOCUS: Detect clearly wrong claims, not micro-validate everything.
        """
        warnings = []

        # Check for obviously wrong acceleration claims
        acc_matches = re.findall(r'(\d+\.?\d*)\s*秒.*?(?:百公里|0-100)', answer)
        for match in acc_matches:
            try:
                acc_time = float(match)
                if acc_time < 1.5 or acc_time > 25:  # Clearly impossible range
                    warnings.append(f"⚠️ 加速时间 {acc_time} 秒看起来不太合理")
            except ValueError:
                continue

        # Check for obviously wrong speeds
        speed_matches = re.findall(r'(?:最高时速|极速).*?(\d+)', answer)
        for match in speed_matches:
            try:
                speed = int(match)
                if speed < 50 or speed > 500:  # Clearly impossible range
                    warnings.append(f"⚠️ 最高时速 {speed} 公里/小时看起来不太合理")
            except ValueError:
                continue

        # Simple context support check - are numbers in answer found in context?
        answer_numbers = re.findall(r'\d+\.?\d*', answer)
        unsupported_numbers = []
        for number in answer_numbers:
            if number not in context:
                unsupported_numbers.append(number)

        if len(unsupported_numbers) > 3:  # Only warn if many numbers are unsupported
            warnings.append("⚠️ 答案中包含较多文档中未提及的数字")

        # Calculate simple quality score
        quality_score = max(0, 100 - len(warnings) * 20)

        return {
            "warnings": warnings,
            "quality_score": quality_score,
            "has_issues": len(warnings) > 0,
            "recommendation": "review_answer" if len(warnings) > 1 else "acceptable"
        }


class LocalLLM:
    """
    SIMPLIFIED Local LLM focused on what actually works for anti-hallucination.

    REMOVED: Complex confidence scoring, over-engineered features
    KEPT: Sentence-level citations, mode-specific parameters, simple fact checking
    """

    def __init__(
            self,
            model_name: Optional[str] = None,
            device: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
    ):
        """Initialize the simplified LLM with environment-driven configuration."""

        # Use environment settings as defaults
        self.model_name = model_name or settings.default_llm_model
        self.model_path = settings.llm_model_full_path
        self.device = device or settings.device
        self.temperature = temperature or settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

        # Quantization settings from environment
        self.use_4bit = settings.llm_use_4bit
        self.use_8bit = settings.llm_use_8bit
        self.torch_dtype = settings.llm_torch_dtype

        # Initialize simplified components
        self.fact_checker = SimpleFactChecker()

        # Log configuration
        print(f"Simplified LocalLLM Configuration:")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Use 4-bit: {self.use_4bit}")
        print(f"  Use 8-bit: {self.use_8bit}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Features: SIMPLIFIED anti-hallucination + sentence citations")

        # Initialize tokenizer and model
        self._load_model()

        # Simplified prompt template with sentence-level citations
        self.qa_prompt_template = self._create_simplified_citation_template()

    def _load_model(self):
        """Load the local LLM model using environment configuration."""
        print(f"Loading LLM model {self.model_path} on {self.device}...")

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

        # Get model loading kwargs from settings
        model_kwargs = settings.get_model_kwargs()

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

    def _create_simplified_citation_template(self) -> str:
        """
        SIMPLIFIED template focused on what works: sentence-level citations.

        REMOVED: Complex mode variations, over-engineered prompts
        KEPT: Clear citation requirements, anti-hallucination focus
        """
        template = """You are a professional automotive expert assistant with STRICT accuracy requirements.

CRITICAL RULES:
1. Only use information explicitly mentioned in the provided documents
2. If specific data is not in documents, say "根据提供文档，未找到具体的[参数]数据"
3. Never estimate, guess, or infer any numerical values
4. MANDATORY: Cite the source document for EVERY factual sentence using 【来源：DOC_X】

SENTENCE-LEVEL CITATION REQUIREMENT:
- Every sentence containing facts must end with 【来源：DOC_X】
- Multiple sources: 【来源：DOC_1, DOC_2】
- Example: "最高时速为220公里/小时【来源：DOC_1】。"

ANTI-HALLUCINATION FOCUS:
- Only state what documents explicitly mention
- If uncertain, acknowledge limitations
- Prioritize documents with higher relevance scores (shown as 🔥, ⭐, 📄)

Document Content:
{context}

Question:
{question}

IMPORTANT: Respond in Chinese with sentence-level citations 【来源：DOC_X】 for every factual statement."""
        return template

    def get_simplified_template_for_mode(self, mode: str) -> str:
        """
        SIMPLIFIED mode-specific templates.

        REMOVED: Complex variations, over-engineered differences
        KEPT: Essential mode differences for facts vs analysis
        """

        base_citation = """
SENTENCE-LEVEL CITATIONS:
- Every factual sentence must end with 【来源：DOC_X】
- Example: "加速时间为3.9秒【来源：DOC_1】。"
"""

        templates = {
            "facts": f"""You are an automotive specifications expert with strict accuracy requirements.

CRITICAL RULES:
1. Only use information explicitly in documents
2. Never estimate or guess numerical values
3. Prioritize high-relevance documents (🔥 > ⭐ > 📄)

{base_citation}

Document Content:
{{context}}

Question:
{{question}}

IMPORTANT: Respond in Chinese with citations 【来源：DOC_X】 for every fact.""",

            "features": f"""You are an automotive product analyst with strict evidence requirements.

ANALYSIS RULES:
1. Base analysis only on document evidence
2. Distinguish between facts and reasoning
3. Cite sources for evidence, mark reasoning as analysis

{base_citation}

Document Content:
{{context}}

Feature Question:
{{question}}

IMPORTANT: Respond in Chinese with citations 【来源：DOC_X】 for evidence.""",

            "quotes": f"""You are an automotive market researcher extracting exact quotes.

EXTRACTION RULES:
1. Only extract quotes that actually exist in documents
2. Use exact quotations - do not modify
3. Never create or fabricate quotes

{base_citation}

Document Content:
{{context}}

Quote Topic:
{{question}}

IMPORTANT: Only extract real quotes with exact citations 【来源：DOC_X】."""
        }

        # Default to facts template for other modes
        return templates.get(mode, templates["facts"])

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
        SIMPLIFIED answer generation with mode-specific parameters.

        FOCUS: What actually works for anti-hallucination.
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

        logger.info(f"LLM inference with mode '{query_mode}': T={temperature}, max_tokens={max_tokens}")

        # Get mode-specific template
        template = self.get_simplified_template_for_mode(query_mode)

        # Format context with relevance scores and token budget
        context_params = mode_config.get_context_params(mode_enum)
        max_context_tokens = context_params["max_context_tokens"]

        context = format_documents_with_relevance_scores(
            documents,
            max_token_budget=max_context_tokens
        )

        # Create prompt
        prompt = template.format(context=context, question=query)

        # Generate with mode-specific parameters
        start_time = time.time()

        try:
            # Use mode-specific parameters
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

            # SIMPLIFIED fact checking
            quality_check = self.fact_checker.simple_quality_check(answer, context)

            # Add simple quality disclaimer if needed
            if quality_check["has_issues"]:
                disclaimer = "\n\n⚠️ 注意: 此答案包含需要验证的信息，建议查阅更多资料确认。"
                answer += disclaimer

            generation_time = time.time() - start_time
            logger.info(f"Mode '{query_mode}' generation completed in {generation_time:.2f}s")
            logger.info(f"Simple quality score: {quality_check['quality_score']:.1f}")

            return answer

        except Exception as e:
            logger.error(f"Generation failed for mode '{query_mode}': {e}")
            raise e

    def simple_confidence_score(
            self,
            answer: str,
            documents: List[Tuple[Document, float]],
            query_mode: str
    ) -> Dict[str, Any]:
        """
        SIMPLIFIED confidence calculation focused on what matters.

        REMOVED: Complex multi-factor scoring, over-engineered metrics
        KEPT: Basic relevance correlation, simple quality assessment
        """
        if not documents:
            return {"confidence": 0, "basis": "no_documents"}

        # Calculate average relevance score
        relevance_scores = [score for _, score in documents]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)

        # Simple quality check
        context = format_documents_with_relevance_scores(documents)
        quality_check = self.fact_checker.simple_quality_check(answer, context)

        # SIMPLIFIED confidence calculation
        base_confidence = min(100, avg_relevance * 100)  # Start with relevance

        # Adjust for quality issues
        if quality_check["has_issues"]:
            base_confidence *= 0.8  # Reduce if quality issues

        # Slight boost for facts mode (more conservative)
        if query_mode == "facts" and avg_relevance > 0.6:
            base_confidence = min(100, base_confidence * 1.1)

        return {
            "confidence": int(base_confidence),
            "avg_relevance": avg_relevance,
            "quality_score": quality_check["quality_score"],
            "has_quality_issues": quality_check["has_issues"],
            "basis": "simplified_relevance_quality"
        }

    def validate_mode(self, mode: str) -> bool:
        """Validate if the query mode is supported."""
        valid_modes = ["facts", "features", "tradeoffs", "scenarios", "debate", "quotes"]
        return mode in valid_modes

    def get_mode_info(self, mode: str) -> Dict[str, Any]:
        """Get simplified information about a specific query mode."""
        mode_info = {
            "facts": {
                "name": "车辆规格查询",
                "description": "直接查询具体的车辆规格参数",
                "complexity": "simple",
                "anti_hallucination": "strict",
                "features": ["sentence_citations", "simple_fact_checking"]
            },
            "features": {
                "name": "功能分析",
                "description": "评估车辆功能和特性",
                "complexity": "moderate",
                "anti_hallucination": "moderate",
                "features": ["sentence_citations", "evidence_based_analysis"]
            },
            "quotes": {
                "name": "用户评论",
                "description": "提取相关的用户评论和反馈",
                "complexity": "simple",
                "anti_hallucination": "strict",
                "features": ["exact_quotes_only", "no_fabrication"]
            }
        }

        # Default info for other modes
        default_info = {
            "name": "通用查询",
            "description": "通用查询模式",
            "complexity": "moderate",
            "anti_hallucination": "moderate",
            "features": ["sentence_citations", "simple_fact_checking"]
        }

        return mode_info.get(mode, default_info)

    def get_model_info(self) -> Dict[str, Any]:
        """Get simplified model information."""
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

        # Simplified model configuration
        model_config = {
            "model_name": self.model_name,
            "device": self.device,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "quantization": "4-bit" if self.use_4bit else "8-bit" if self.use_8bit else "none",
            "environment_driven": True,
            "worker_type": os.environ.get("WORKER_TYPE", "unknown"),
            "memory_fraction": settings.get_worker_memory_fraction(),
            "tesla_t4_optimized": not self.use_4bit,

            # SIMPLIFIED features
            "anti_hallucination_approach": "simplified_effective",
            "default_mode": "facts",
            "response_language": "chinese",
            "template_language": "english",
            "supported_modes": ["facts", "features", "tradeoffs", "scenarios", "debate", "quotes"],

            "key_features": {
                "sentence_level_citations": True,
                "relevance_score_display": True,
                "simple_fact_checking": True,
                "mode_specific_parameters": True,
                "token_budget_management": True,
                "simplified_confidence": True,
                # REMOVED complex features
                "complex_confidence_scoring": False,
                "contradiction_detection": False,
                "structured_json_output": False,
                "multi_phase_filtering": False
            }
        }

        return {**model_config, **memory_info}