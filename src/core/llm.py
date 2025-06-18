import json
import time
import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import jieba
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from src.config.settings import settings

# Import shared utilities
from src.utils.quality_utils import (
    extract_automotive_key_phrases,
    check_acceleration_claims,
    check_numerical_specs_realistic,
    has_numerical_data
)

logger = logging.getLogger(__name__)


class AutomotiveFactChecker:
    """
    Fact checker for automotive specifications to detect obvious hallucinations.
    Uses shared utility functions to avoid duplication.
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

        Returns:
            Dictionary with warnings and quality score
        """
        warnings = []

        # Use shared utility functions
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
        """Check if numerical claims in answer are supported by context."""
        import re
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
    Uses shared utility functions to avoid duplication.
    """

    def __init__(self):
        self.fact_checker = AutomotiveFactChecker()

    def calculate_confidence(self, answer: str, context: str, documents: List[Tuple[Document, float]]) -> Dict[
        str, any]:
        """
        Calculate comprehensive confidence score for an answer.

        Returns:
            Dictionary with confidence metrics and recommendations
        """
        scores = {}

        # 1. Context Support Score (0-100)
        scores['context_support'] = self._calculate_context_support(answer, context)

        # 2. Document Relevance Score (0-100)
        scores['document_relevance'] = self._calculate_document_relevance(answer, documents)

        # 3. Factual Consistency Score (0-100)
        scores['factual_consistency'] = self._calculate_factual_consistency(answer, context)

        # 4. Specificity Score (0-100)
        scores['specificity'] = self._calculate_specificity(answer)

        # 5. Uncertainty Indicators (0-100, higher = more uncertain)
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

        # Use shared utility to extract key phrases
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
        """Calculate how specific the answer is for Chinese text (specific answers are generally more reliable)."""
        import re
        specificity_indicators = 0

        # Check for specific numbers
        if re.search(r'\d+\.?\d*', answer):
            specificity_indicators += 1

        # Check for Chinese automotive units and terms
        unit_patterns = [
            r'(?:秒|升|L|马力|HP|牛米|Nm|公里|km|米|m|毫米|mm|公斤|kg|元|万元)',
            r'(?:公里/小时|km/h|千瓦|kW|立方|吨|分钟|小时)',
            r'(?:百公里加速|油耗|续航|扭矩|功率|排量|轴距)'
        ]
        for pattern in unit_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                specificity_indicators += 1
                break

        # Check for Chinese car brand/model names
        chinese_brands = [
            '宝马', '奔驰', '奥迪', '丰田', '本田', '大众', '特斯拉',
            '福特', '雪佛兰', '日产', '现代', '起亚', '斯巴鲁', '马自达',
            '沃尔沃', '捷豹', '路虎', '雷克萨斯', '讴歌', '英菲尼迪',
            '凯迪拉克', '吉普', '法拉利', '兰博基尼', '保时捷',
            '比亚迪', '蔚来', '理想', '小鹏', '哪吒', '零跑',
            '吉利', '长城', '奇瑞', '长安', '广汽', '一汽'
        ]
        if any(brand in answer for brand in chinese_brands):
            specificity_indicators += 1

        # Check for year mentions (Chinese format)
        if re.search(r'(?:20\d{2}|19\d{2})年?', answer):
            specificity_indicators += 1

        # Normalize to 0-100
        max_indicators = 4
        return (specificity_indicators / max_indicators) * 100

    def _detect_uncertainty_indicators(self, answer: str) -> float:
        """Detect uncertainty indicators in Chinese answers."""
        # Enhanced Chinese uncertainty phrases
        uncertainty_phrases = [
            '可能', '大概', '估计', '应该', '似乎', '看起来', '据说',
            '大致', '约', '左右', '差不多', '基本上', '一般来说',
            '通常', '可能是', '或许', '也许', '估算', '预计',
            '疑似', '推测', '猜测', '不确定', '不清楚', '不详',
            '可能会', '应该是', '看上去', '听说', '传说',
            # English uncertainty phrases (just in case)
            'maybe', 'probably', 'likely', 'appears', 'seems', 'roughly',
            'approximately', 'about', 'around', 'possibly', 'perhaps'
        ]

        uncertainty_count = 0
        for phrase in uncertainty_phrases:
            uncertainty_count += answer.lower().count(phrase.lower())

        # Normalize to 0-100 (higher = more uncertain)
        max_uncertainty = 5  # If more than 5 uncertainty indicators, max score
        return min(100, (uncertainty_count / max_uncertainty) * 100)

    def _generate_recommendation(self, overall_confidence: float, scores: Dict[str, float]) -> str:
        """Generate actionable recommendation based on confidence scores."""
        if overall_confidence >= 85:
            return "高置信度答案，可以直接使用"
        elif overall_confidence >= 70:
            return "中等置信度，建议进行人工验证"
        elif overall_confidence >= 50:
            return "低置信度，需要额外验证和来源确认"
        else:
            return "极低置信度，可能存在错误，建议重新查询"

    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level label."""
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

    UNIFIED SYSTEM: Only supports enhanced queries with mode-specific templates.
    Facts mode serves as the default and replaces old normal queries.
    Tesla T4 optimized with proper quantization handling.
    Enhanced with anti-hallucination features.
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

        All parameters are optional and will use environment settings by default.
        This ensures consistent Tesla T4 optimization across the entire system.
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
        print(f"LocalLLM Configuration (Unified System with Anti-Hallucination):")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Use 4-bit: {self.use_4bit}")
        print(f"  Use 8-bit: {self.use_8bit}")
        print(f"  Torch dtype: {self.torch_dtype}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Query System: UNIFIED (Enhanced Only)")
        print(f"  Default Mode: FACTS")
        print(f"  Anti-Hallucination: ENABLED")

        # Initialize tokenizer and model
        self._load_model()

        # Enhanced QA prompt template with anti-hallucination measures
        self.qa_prompt_template = self._create_anti_hallucination_qa_prompt_template()

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

    def _create_anti_hallucination_qa_prompt_template(self) -> str:
        """
        Create an enhanced QA prompt template with strong anti-hallucination measures.
        All responses must be in Chinese.
        """
        template = """你是一位专业的汽车规格专家助手，具有严格的准确性要求。

关键规则：
1. 只能使用提供的文档中明确提到的信息
2. 如果文档中没有提及具体的数字/规格，请说"根据提供的文档，未找到具体的[参数名称]数据"
3. 绝对不要估计、猜测或推断任何数值
4. 如果文档内容不清楚或有矛盾，请承认这种不确定性
5. 始终引用找到信息的确切来源

数值准确性检查：
- 百公里加速：正常范围是3-15秒
- 如果看到明显错误的数值（如0.8秒），请标注为可疑
- 始终根据汽车标准对技术规格进行双重检查

你的任务是帮助用户查找汽车规格、功能和技术细节的信息。

只能使用以下文档内容回答问题。如果文档中没有答案，请说明你不知道，并建议需要什么额外信息。

文档内容：
{context}

问题：
{question}

回答格式：
1. 基于文档的直接答案（或"未找到相关信息"）
2. 来源引用
3. 如果不确定，明确说明限制

请用中文回答，并引用找到信息的具体来源（文档标题或网址）。"""
        return template

    def get_prompt_template_for_mode(self, mode: str) -> str:
        """
        Get specialized prompt template for different query modes.
        All templates ensure Chinese responses with anti-hallucination measures.
        """

        templates = {
            "facts": """你是一位专业的汽车规格专家助手，具有严格的准确性要求。

关键规则：
1. 只能使用提供的文档中明确提到的信息
2. 如果文档中没有提及具体的数字/规格，请说"根据提供的文档，未找到具体的[参数名称]数据"
3. 绝对不要估计、猜测或推断任何数值
4. 如果文档内容不清楚或有矛盾，请承认这种不确定性
5. 始终引用找到信息的确切来源

数值准确性检查：
- 百公里加速：正常范围是3-15秒
- 如果看到明显错误的数值（如0.8秒），请标注为可疑
- 始终根据汽车标准对技术规格进行双重检查

只能使用以下文档内容回答问题。如果文档中没有答案，请说明你不知道，并建议需要什么额外信息。

文档内容：
{context}

问题：
{question}

请用中文回答，并引用找到信息的具体来源（文档标题或网址）。""",

            "features": """你是一位专业的汽车产品策略专家助手，具有严格的准确性要求。

关键规则：
1. 只能使用提供的文档中明确提到的信息
2. 分析必须基于文档中找到的证据
3. 绝对不要做出超出文档内容的假设
4. 如果文档缺乏相关信息，请明确说明这一限制

你的任务是分析是否应该添加某项功能，严格基于提供的文档内容。

请分两个部分分析功能需求：
【实证分析】 - 基于提供文档的实证分析
【策略推理】 - 基于找到证据的策略推理

要实事求是并引用具体来源。不要做出超出文档内容的假设。

文档内容：
{context}

功能问题：
{question}

请用中文提供分析，并引用找到信息的具体来源（文档标题或网址）。""",

            "tradeoffs": """你是一位专业的汽车设计决策分析师，具有严格的准确性要求。

关键规则：
1. 只能使用提供的文档中明确提到的信息
2. 优缺点分析必须基于文档证据
3. 绝对不要推测超出文档内容的情况
4. 如果文档缺乏足够的比较信息，请明确说明

你的任务是分析设计选择的优缺点，严格基于提供的文档内容。

请分两个部分分析：
【文档支撑】 - 来自提供文档的证据
【利弊分析】 - 基于证据的优缺点分析

要客观并引用具体来源。不要推测超出文档内容的情况。

文档内容：
{context}

设计决策问题：
{question}

请用中文提供分析，并引用找到信息的具体来源（文档标题或网址）。""",

            "scenarios": """你是一位专业的汽车用户体验分析师，具有严格的准确性要求。

关键规则：
1. 只能使用提供的文档中明确提到的信息
2. 场景分析必须基于文档证据
3. 绝对不要创造文档中未提及的场景
4. 如果文档缺乏相关场景信息，请明确说明

你的任务是分析功能在真实使用场景中的表现，严格基于提供的文档内容。

请分两个部分分析：
【文档场景】 - 提供文档中提及的场景
【场景推理】 - 基于找到证据的场景分析

要具体并引用来源。不要创造文档中未提及的场景。

文档内容：
{context}

场景问题：
{question}

请用中文提供分析，并引用找到信息的具体来源（文档标题或网址）。""",

            "debate": """你是一位专业的汽车行业圆桌讨论主持人，具有严格的准确性要求。

关键规则：
1. 只能使用提供的文档中明确提到的信息
2. 观点必须基于文档中找到的证据
3. 绝对不要编造文档中不支持的观点
4. 如果文档缺乏足够的多角度分析信息，请明确说明

你的任务是基于提供的文档内容呈现不同专业角度的观点。

请呈现以下角度的观点：
**👔 产品经理角度：** 基于文档中的证据
**🔧 工程师角度：** 基于文档中的技术信息
**👥 用户代表角度：** 基于文档中的用户反馈

**📋 讨论总结：** 仅综合文档可以支持的内容

要实事求是并为每个角度引用具体来源。

文档内容：
{context}

讨论话题：
{question}

请用中文提供观点，并引用找到信息的具体来源（文档标题或网址）。""",

            "quotes": """你是一位专业的汽车市场研究分析师，具有严格的准确性要求。

关键规则：
1. 只提取提供文档中实际存在的引用
2. 使用确切的引文 - 不要改写或修改
3. 绝对不要创造或编造引文
4. 如果找不到相关引文，请明确说明

你的任务是从提供的文档内容中提取实际的用户引文和反馈。

请按以下格式提取引文：
【来源1】："文档中的确切引文..."
【来源2】："文档中的另一个确切引文..."

如果找不到相关的用户引文，请说明："根据提供的文档，未找到相关的用户评论或反馈。"

关键：只提取文档中实际存在的引文。不要创造或改写内容。

文档内容：
{context}

引文话题：
{question}

请用中文提供引文，并引用找到它们的具体来源（文档标题或网址）。"""
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

        Args:
            query: The user's query
            documents: Retrieved documents with scores
            query_mode: The query mode to use (defaults to "facts")
            metadata_filter: Optional metadata filters

        Returns:
            Generated answer using the mode-specific template with fact checking
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

            # Perform fact checking
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
        """Create a strict prompt for answer regeneration in Chinese."""
        warnings_text = "\n".join(f"- {warning}" for warning in warnings)

        prompt = f"""作为汽车规格专家，请基于以下文档回答问题。

关键：之前的回答检测到以下问题：
{warnings_text}

严格要求：
1. 只使用文档中明确提到的信息
2. 如果文档中没有具体数据，明确说明"文档中未提及此数据"
3. 不要猜测或推断任何数值
4. 如果发现不合理的数据，请质疑其准确性
5. 所有数值必须能在文档中找到对应内容

文档内容：
{context}

问题：{query}

请提供准确、有依据的中文答案，并引用具体来源："""

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

        Returns:
            Dictionary with answer, confidence metrics, and recommendations
        """
        # Generate the answer with anti-hallucination measures
        answer = self._answer_with_anti_hallucination(query, documents, query_mode, metadata_filter)

        # Calculate confidence
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
        """
        Validate if the query mode is supported.

        Args:
            mode: Query mode to validate

        Returns:
            True if mode is valid, False otherwise
        """
        valid_modes = ["facts", "features", "tradeoffs", "scenarios", "debate", "quotes"]
        return mode in valid_modes

    def get_mode_info(self, mode: str) -> Dict[str, Any]:
        """
        Get information about a specific query mode.

        Args:
            mode: Query mode

        Returns:
            Dictionary with mode information
        """
        mode_info = {
            "facts": {
                "name": "车辆规格查询",
                "description": "直接验证具体的车辆规格参数",
                "two_layer": False,
                "complexity": "simple",
                "template_type": "anti_hallucination_qa",
                "is_default": True,
                "anti_hallucination": True
            },
            "features": {
                "name": "新功能建议",
                "description": "评估是否应该添加某项功能",
                "two_layer": True,
                "complexity": "moderate",
                "template_type": "structured_analysis_with_fact_check",
                "is_default": False,
                "anti_hallucination": True
            },
            "tradeoffs": {
                "name": "权衡利弊分析",
                "description": "分析设计选择的优缺点",
                "two_layer": True,
                "complexity": "complex",
                "template_type": "structured_analysis_with_fact_check",
                "is_default": False,
                "anti_hallucination": True
            },
            "scenarios": {
                "name": "用户场景分析",
                "description": "评估功能在实际使用场景中的表现",
                "two_layer": True,
                "complexity": "complex",
                "template_type": "structured_analysis_with_fact_check",
                "is_default": False,
                "anti_hallucination": True
            },
            "debate": {
                "name": "多角色讨论",
                "description": "模拟不同角色的观点和讨论",
                "two_layer": False,
                "complexity": "complex",
                "template_type": "multi_perspective_with_fact_check",
                "is_default": False,
                "anti_hallucination": True
            },
            "quotes": {
                "name": "原始用户评论",
                "description": "提取相关的用户评论和反馈",
                "two_layer": False,
                "complexity": "simple",
                "template_type": "extraction_with_verification",
                "is_default": False,
                "anti_hallucination": True
            }
        }

        return mode_info.get(mode, mode_info["facts"])

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model including environment config and anti-hallucination features."""
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

        # Model configuration info including environment settings and anti-hallucination features
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
            "query_system": "unified_enhanced_with_anti_hallucination",
            "default_mode": "facts",
            "template_system": "anti_hallucination_enhanced",
            "supported_modes": ["facts", "features", "tradeoffs", "scenarios", "debate", "quotes"],
            "anti_hallucination_features": {
                "fact_checker": True,
                "confidence_scorer": True,
                "strict_prompts": True,
                "context_verification": True,
                "numerical_validation": True,
                "regeneration_on_issues": True
            }
        }

        return {**model_config, **memory_info}