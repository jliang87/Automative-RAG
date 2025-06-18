import json
import time
import os
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

import torch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from src.config.settings import settings

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


class LocalLLM:
    """
    Local DeepSeek LLM integration for RAG with GPU acceleration.

    UNIFIED SYSTEM: Only supports enhanced queries with mode-specific templates.
    Facts mode serves as the default and replaces old normal queries.
    Tesla T4 optimized with proper quantization handling.
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

        # Log configuration for debugging
        print(f"LocalLLM Configuration (Unified System):")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Use 4-bit: {self.use_4bit}")
        print(f"  Use 8-bit: {self.use_8bit}")
        print(f"  Torch dtype: {self.torch_dtype}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Query System: UNIFIED (Enhanced Only)")
        print(f"  Default Mode: FACTS")

        # Initialize tokenizer and model
        self._load_model()

        # RESTORED: Original QA prompt template for Facts mode
        self.qa_prompt_template = self._create_original_qa_prompt_template()

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

    def _create_original_qa_prompt_template(self) -> str:
        """
        RESTORED: Create the original QA prompt template that was working.
        This is used for Facts mode to ensure it works exactly like the old system.
        """
        template = """You are an automotive specifications expert assistant.

Your task is to help users find information about automotive specifications, features, and technical details.

Use ONLY the following context to answer the question. If the context doesn't contain the answer, say you don't know and suggest what additional information might be needed.

Be concise, clear, and factual. Focus on providing accurate technical information. Highlight key specifications like horsepower, torque, dimensions, fuel efficiency, etc. when relevant.

Context:
{context}

Question:
{question}

When providing your answer, cite the specific sources (document titles or URLs) where you found the information."""
        return template

    def get_prompt_template_for_mode(self, mode: str) -> str:
        """
        Get specialized prompt template for different query modes.

        FIXED: All modes now use the same strict, factual approach as Facts mode.
        This ensures consistent quality and accuracy across all query types.
        """

        templates = {
            "facts": """You are an automotive specifications expert assistant.

Your task is to help users find information about automotive specifications, features, and technical details.

Use ONLY the following context to answer the question. If the context doesn't contain the answer, say you don't know and suggest what additional information might be needed.

Be concise, clear, and factual. Focus on providing accurate technical information. Highlight key specifications like horsepower, torque, dimensions, fuel efficiency, etc. when relevant.

Context:
{context}

Question:
{question}

When providing your answer, cite the specific sources (document titles or URLs) where you found the information.""",

            "features": """You are an automotive product strategy expert assistant.

Your task is to analyze whether a specific feature should be added, based strictly on the provided context.

Use ONLY the following context to answer the question. If the context doesn't contain relevant information about the feature, say you don't know and suggest what additional information might be needed.

Analyze the feature request in two sections:
【实证分析】 - Evidence-based analysis from the provided documents
【策略推理】 - Strategic reasoning based on the evidence found

Be factual and cite specific sources. Do not make assumptions beyond what the context provides.

Context:
{context}

Feature Question:
{question}

When providing your analysis, cite the specific sources (document titles or URLs) where you found the information.""",

            "tradeoffs": """You are an automotive design decision analyst.

Your task is to analyze the pros and cons of design choices based strictly on the provided context.

Use ONLY the following context to answer the question. If the context doesn't contain sufficient information for comparison, say you don't know and suggest what additional information might be needed.

Analyze in two sections:
【文档支撑】 - Evidence from the provided documents  
【利弊分析】 - Pros and cons analysis based on the evidence

Be objective and cite specific sources. Do not speculate beyond what the context provides.

Context:
{context}

Design Decision Question:
{question}

When providing your analysis, cite the specific sources (document titles or URLs) where you found the information.""",

            "scenarios": """You are an automotive user experience analyst.

Your task is to analyze how features perform in real-world scenarios based strictly on the provided context.

Use ONLY the following context to answer the question. If the context doesn't contain relevant scenario information, say you don't know and suggest what additional information might be needed.

Analyze in two sections:
【文档场景】 - Scenarios mentioned in the provided documents
【场景推理】 - Scenario analysis based on the evidence found

Be specific and cite sources. Do not create scenarios not mentioned in the context.

Context:
{context}

Scenario Question:
{question}

When providing your analysis, cite the specific sources (document titles or URLs) where you found the information.""",

            "debate": """You are an automotive industry roundtable moderator.

Your task is to present different professional perspectives based strictly on the provided context.

Use ONLY the following context to answer the question. If the context doesn't contain enough information for multi-perspective analysis, say you don't know and suggest what additional information might be needed.

Present viewpoints from:
**👔 Product Manager Perspective:** Based on evidence in the context
**🔧 Engineer Perspective:** Based on technical information in the context  
**👥 User Representative Perspective:** Based on user feedback in the context

**📋 Discussion Summary:** Synthesize only what can be supported by the context

Be factual and cite specific sources for each perspective.

Context:
{context}

Discussion Topic:
{question}

When providing perspectives, cite the specific sources (document titles or URLs) where you found the information.""",

            "quotes": """You are an automotive market research analyst.

Your task is to extract actual user quotes and feedback from the provided context.

Use ONLY the following context to find user comments. If the context doesn't contain user quotes or feedback, say you don't know and suggest what additional information might be needed.

Extract quotes in this format:
【来源1】："Exact quote from the document..."
【来源2】："Another exact quote from the document..."

If no relevant user quotes are found, state: "根据提供的文档，未找到相关的用户评论或反馈。"

CRITICAL: Only extract quotes that actually exist in the provided context. Do not create or paraphrase content.

Context:
{context}

Quote Topic:
{question}

When providing quotes, cite the specific sources (document titles or URLs) where you found them."""
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
        UNIFIED: Answer a query using a specific mode template.

        FIXED: All modes now use the same proven approach as Facts mode.

        Args:
            query: The user's query
            documents: Retrieved documents with scores
            query_mode: The query mode to use (defaults to "facts")
            metadata_filter: Optional metadata filters

        Returns:
            Generated answer using the mode-specific template
        """
        # Validate mode (fallback to facts)
        if not self.validate_mode(query_mode):
            logger.warning(f"Invalid query mode '{query_mode}', using facts mode")
            query_mode = "facts"

        # FIXED: All modes now use the same proven generation approach
        return self._answer_with_proven_approach(query, documents, query_mode, metadata_filter)

    def _answer_with_proven_approach(
            self,
            query: str,
            documents: List[Tuple[Document, float]],
            query_mode: str,
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
    ) -> str:
        """
        FIXED: Use the proven working approach for ALL modes.
        This ensures consistent quality and performance across all query types.
        """
        # Get the appropriate template for this mode
        template = self.get_prompt_template_for_mode(query_mode)

        # Format documents into context using the same proven method
        context = _format_documents_for_context(documents)

        # Create prompt using the mode-specific template
        prompt = template.format(
            context=context,
            question=query
        )

        # Generate answer using the SAME proven parameters as Facts mode
        start_time = time.time()

        try:
            # Use consistent generation parameters (same as the working Facts mode)
            results = self.pipe(
                prompt,
                num_return_sequences=1,
                do_sample=True,
                temperature=self.temperature,  # Use the proven temperature
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_tokens  # Use consistent token count
            )

            generation_time = time.time() - start_time
            print(f"Mode '{query_mode}' answer generated in {generation_time:.2f} seconds")

            answer = results[0]["generated_text"]
            return answer

        except Exception as e:
            print(f"❌ Generation failed for mode '{query_mode}': {e}")
            if "CUDA" in str(e):
                print("This may be a Tesla T4 memory or quantization issue.")
                print("Check your environment settings:")
                print(f"  LLM_USE_4BIT: {settings.llm_use_4bit}")
                print(f"  GPU_MEMORY_FRACTION_INFERENCE: {settings.gpu_memory_fraction_inference}")
            raise e

    def _answer_facts_mode_original(
            self,
            query: str,
            documents: List[Tuple[Document, float]],
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
    ) -> str:
        """
        RESTORED: Use the original working logic for Facts mode.
        This ensures Facts mode works exactly like the old basic query system.
        """
        # Use the original QA template
        context = _format_documents_for_context(documents)

        prompt = self.qa_prompt_template.format(
            context=context,
            question=query
        )

        # Generate answer using the same approach as the old system
        start_time = time.time()

        try:
            results = self.pipe(
                prompt,
                num_return_sequences=1,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_tokens
            )

            generation_time = time.time() - start_time
            print(f"Facts mode (original) answer generated in {generation_time:.2f} seconds")

            answer = results[0]["generated_text"]
            return answer

        except Exception as e:
            print(f"❌ Facts mode generation failed: {e}")
            raise e

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
                "two_layer": False,  # UNIFIED: Facts mode is direct
                "complexity": "simple",
                "template_type": "original_qa",  # FIXED: Uses original template
                "is_default": True  # NEW: Indicates this is the default mode
            },
            "features": {
                "name": "新功能建议",
                "description": "评估是否应该添加某项功能",
                "two_layer": True,
                "complexity": "moderate",
                "template_type": "structured_analysis",
                "is_default": False
            },
            "tradeoffs": {
                "name": "权衡利弊分析",
                "description": "分析设计选择的优缺点",
                "two_layer": True,
                "complexity": "complex",
                "template_type": "structured_analysis",
                "is_default": False
            },
            "scenarios": {
                "name": "用户场景分析",
                "description": "评估功能在实际使用场景中的表现",
                "two_layer": True,
                "complexity": "complex",
                "template_type": "structured_analysis",
                "is_default": False
            },
            "debate": {
                "name": "多角色讨论",
                "description": "模拟不同角色的观点和讨论",
                "two_layer": False,
                "complexity": "complex",
                "template_type": "multi_perspective",
                "is_default": False
            },
            "quotes": {
                "name": "原始用户评论",
                "description": "提取相关的用户评论和反馈",
                "two_layer": False,
                "complexity": "simple",
                "template_type": "extraction",
                "is_default": False
            }
        }

        return mode_info.get(mode, mode_info["facts"])

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model including environment config."""
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

        # Model configuration info including environment settings
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
            "tesla_t4_optimized": not self.use_4bit,  # True if 4-bit is disabled
            "query_system": "unified_enhanced",  # UNIFIED: Only enhanced queries
            "default_mode": "facts",  # NEW: Default query mode
            "template_system": "hybrid_original_enhanced",  # FIXED: Indicates Facts uses original
            "supported_modes": ["facts", "features", "tradeoffs", "scenarios", "debate", "quotes"]
        }

        return {**model_config, **memory_info}