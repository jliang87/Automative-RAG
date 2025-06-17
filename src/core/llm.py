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

    def get_prompt_template_for_mode(self, mode: str) -> str:
        """Get specialized prompt template for different query modes."""

        templates = {
            "facts": """你是专业的汽车技术规格验证专家。请严格按照以下格式回答：

    【实证答案】
    基于提供的文档内容，回答用户的具体问题。如果文档中没有相关信息，必须明确说明"根据提供的文档，未提及相关信息"。

    【推理补充】
    如果有需要，可以基于汽车行业常识进行合理推测，但必须清楚标注这是推理而非文档事实。

    提供的文档内容：
    {context}

    用户问题：
    {question}

    请确保回答精确、可验证，并明确区分事实和推理。""",

            "features": """你是汽车产品策略专家。请按照以下格式分析是否应该添加某项功能：

    【实证分析】
    基于提供的文档中关于类似功能或相关技术的信息进行分析。如果文档中没有相关信息，说明"根据提供的文档，未找到相关功能信息"。

    【策略推理】
    基于产品思维和用户需求，分析这个功能的潜在价值：
    - 用户受益分析：谁会从这个功能中受益？
    - 技术可行性：实现难度和技术要求
    - 市场竞争优势：相比竞品的差异化价值
    - 成本效益评估：投入产出比分析

    提供的文档内容：
    {context}

    用户询问的功能：
    {question}

    请提供平衡、专业的评估意见。""",

            "tradeoffs": """你是汽车设计决策分析师。请按照以下格式分析设计选择的利弊：

    【文档支撑】
    基于提供文档中的相关信息和数据进行分析。如果文档中缺少信息，明确说明。

    【利弊分析】
    **优点：**
    - [基于文档的优点]
    - [基于行业经验推理的优点]

    **缺点：**
    - [基于文档的缺点]
    - [基于行业经验推理的缺点]

    **总结建议：**
    综合评估和具体建议

    提供的文档内容：
    {context}

    设计决策问题：
    {question}

    请确保分析客观、全面，区分事实和推理。""",

            "scenarios": """你是用户体验分析专家。请按照以下格式分析功能在不同场景下的表现：

    【文档场景】
    提取文档中提到的使用场景、用户反馈和实际应用案例。

    【场景推理】
    基于产品思维和用户同理心，分析在以下维度的表现：
    - 目标用户群：谁会最需要这个功能？
    - 使用时机：什么时候这个功能最有价值？
    - 最佳条件：在什么条件下效果最好？
    - 潜在问题：可能遇到的限制和挑战
    - 改进建议：如何优化用户体验

    提供的文档内容：
    {context}

    分析主题：
    {question}

    请提供具体、实用的场景分析，重点关注用户实际需求。""",

            "debate": """你是汽车行业圆桌讨论主持人。请模拟以下三个角色对这个问题的不同观点：

    **👔 产品经理观点：**
    从商业价值、市场需求、用户体验和产品策略角度分析

    **🔧 工程师观点：**
    从技术实现难度、成本控制、系统集成和可靠性角度分析

    **👥 用户代表观点：**
    从实际使用需求、日常体验、价格敏感度和功能实用性角度分析

    **📋 讨论总结：**
    - 共同观点：三方都认同的点
    - 主要分歧：存在不同看法的地方
    - 平衡建议：综合考虑的解决方案

    提供的文档内容：
    {context}

    讨论话题：
    {question}

    请让每个角色基于各自专业背景提出有深度的观点。""",

            "quotes": """你是汽车市场研究分析师。请从提供的文档中提取与查询主题相关的用户原始评论和反馈：

    请严格按以下格式提供用户评论，只使用文档中的真实内容：

    【来源1】："这里是文档中的原始用户评论或反馈..."
    【来源2】："这里是另一条文档中的原始评论..."
    【来源3】："这里是第三条相关的用户反馈..."

    如果文档中没有找到相关的用户评论，请明确说明："根据提供的文档，未找到相关的用户评论或反馈。"

    提供的文档内容：
    {context}

    查询主题：
    {question}

    重要：只提取真实存在于文档中的用户评论，不要编造或推测内容。"""
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
        Answer a query using a specific mode template.

        Args:
            query: The user's query
            documents: Retrieved documents with scores
            query_mode: The query mode to use
            metadata_filter: Optional metadata filters

        Returns:
            Generated answer using the mode-specific template
        """
        # Format documents into context
        context = _format_documents_for_context(documents)

        # Get the appropriate prompt template
        template = self.get_prompt_template_for_mode(query_mode)

        # Create prompt using the mode-specific template
        prompt = template.format(
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
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_tokens * 2 if query_mode in ["debate", "scenarios",
                                                                     "tradeoffs"] else self.max_tokens
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