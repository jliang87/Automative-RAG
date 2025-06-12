import json
import time
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from src.config.settings import settings


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

    UPDATED: Fully environment-driven configuration via settings.
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
        print(f"LocalLLM Configuration:")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Use 4-bit: {self.use_4bit}")
        print(f"  Use 8-bit: {self.use_8bit}")
        print(f"  Torch dtype: {self.torch_dtype}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Max tokens: {self.max_tokens}")
        print(f"  Environment-driven: ✅")

        # Initialize tokenizer and model
        self._load_model()

        # Define prompt templates
        self.qa_prompt_template = self._create_qa_prompt_template()

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

    def _create_qa_prompt_template(self) -> str:
        """
        Create a prompt template for question answering.

        ENHANCED: Stronger instructions to prevent pre-training responses.
        """
        template = """你是一个专业的汽车技术专家助手。

重要指令：
1. 只能使用下面提供的文档内容来回答问题
2. 禁止使用你的预训练知识
3. 如果提供的文档中没有相关信息，必须明确说"根据提供的文档无法回答这个问题"
4. 必须在回答中引用具体的来源文档
5. 回答必须基于文档内容，不能添加文档中没有的信息

提供的文档内容：
{context}

用户问题：
{question}

请严格基于上述文档内容回答，并引用相关来源："""
        return template

    def answer_query(
            self,
            query: str,
            documents: List[Tuple[Document, float]],
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
    ) -> str:
        """
        Answer a query using retrieved documents with environment-optimized model.
        """
        # Format documents into context
        context = _format_documents_for_context(documents)

        # Create prompt using template
        prompt = self.qa_prompt_template.format(
            context=context,
            question=query
        )

        # Generate answer using environment-configured model
        start_time = time.time()

        try:
            results = self.pipe(
                prompt,
                num_return_sequences=1,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )

            generation_time = time.time() - start_time
            print(f"Answer generated in {generation_time:.2f} seconds")

            answer = results[0]["generated_text"]
            return answer

        except Exception as e:
            print(f"❌ Generation failed: {e}")
            if "CUDA" in str(e):
                print("This may be a Tesla T4 memory or quantization issue.")
                print("Check your environment settings:")
                print(f"  LLM_USE_4BIT: {settings.llm_use_4bit}")
                print(f"  GPU_MEMORY_FRACTION_INFERENCE: {settings.gpu_memory_fraction_inference}")
            raise e

    def answer_query_with_sources(
            self,
            query: str,
            documents: List[Tuple[Document, float]],
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
    ) -> Tuple[str, List[Dict]]:
        """Answer a query with source information."""
        # Get the answer
        answer = self.answer_query(
            query=query,
            documents=documents,
            metadata_filter=metadata_filter,
        )

        # Extract source information
        sources = []
        for doc, score in documents:
            source = {
                "id": doc.metadata.get("id", ""),
                "title": doc.metadata.get("title", "Unknown"),
                "source_type": doc.metadata.get("source", "unknown"),
                "url": doc.metadata.get("url"),
                "relevance_score": score,
            }
            sources.append(source)

        return answer, sources

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
        }

        return {**model_config, **memory_info}