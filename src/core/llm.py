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


class LocalDeepSeekLLM:
    """
    Local DeepSeek LLM integration for RAG with GPU acceleration.

    This class handles:
    1. Formatting contexts from retrieved documents
    2. Creating prompts with automotive context
    3. Generating responses with source attribution using local model
    4. Loading models from local paths or Hugging Face
    """

    def __init__(
            self,
            model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            device: Optional[str] = None,
            temperature: float = 0.1,
            max_tokens: int = 512,
            use_4bit: bool = True,
            use_8bit: bool = False,
            torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the local DeepSeek LLM with GPU support.

        Args:
            model_name: Name or path of the DeepSeek model
            device: Device to run the model on (cuda:0 or cpu)
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            use_4bit: Whether to use 4-bit quantization (saves GPU memory)
            use_8bit: Whether to use 8-bit quantization (alternative to 4-bit)
            torch_dtype: Torch data type (defaults to float16 for GPU)
        """
        # Get the complete model path if it's a local path
        from src.utils.model_paths import get_llm_model_path
        if not model_name.startswith("http") and "/" in model_name:
            self.model_name = get_llm_model_path(model_name)
        else:
            self.model_name = model_name

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_4bit = use_4bit and self.device.startswith("cuda")
        self.use_8bit = use_8bit and self.device.startswith("cuda") and not self.use_4bit

        # Set default torch dtype based on device
        if torch_dtype is None:
            self.torch_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        else:
            self.torch_dtype = torch_dtype

        # Initialize tokenizer and model
        self._load_model()

        # Define prompt templates
        self.qa_prompt_template = self._create_qa_prompt_template()

    def _load_model(self):
        """Load the local DeepSeek model with appropriate configuration."""
        print(f"Loading DeepSeek model {self.model_name} on {self.device}...")

        # Start timing
        start_time = time.time()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=True  # Only use local files, don't try to download
        )

        # Configure quantization
        if self.use_4bit:
            print("Using 4-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif self.use_8bit:
            print("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
            local_files_only=True  # Only use local files, don't try to download
        )

        # Create generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            repetition_penalty=1.1
        )

        # Report loading time
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds.")

    def _create_qa_prompt_template(self) -> str:
        """
        Create a prompt template for question answering.

        Returns:
            String template for QA
        """
        template = """You are an automotive specifications expert assistant.

Your task is to help users find information about automotive specifications, features, and technical details.

Use ONLY the following context to answer the question. If the context doesn't contain the answer, say you don't know and suggest what additional information might be needed.

Be concise, clear, and factual. Focus on providing accurate technical information. Highlight key specifications like horsepower, torque, dimensions, fuel efficiency, etc. when relevant.

Context:
{context}

Question:
{question}

When providing your answer, cite the specific sources (document titles or URLs) where you found the information.
"""
        return template

    def _format_documents_for_context(
            self, documents: List[Tuple[Document, float]]
    ) -> str:
        """
        Format retrieved documents into context for the prompt.

        Args:
            documents: List of (document, score) tuples

        Returns:
            Formatted context string
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

    def answer_query(
            self,
            query: str,
            documents: List[Tuple[Document, float]],
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
    ) -> str:
        """
        Answer a query using retrieved documents with local DeepSeek model.

        Args:
            query: User query
            documents: List of (document, score) tuples
            metadata_filter: Optional metadata filters used

        Returns:
            Generated answer
        """
        # Format documents into context
        context = self._format_documents_for_context(documents)

        # Create prompt using template
        prompt = self.qa_prompt_template.format(
            context=context,
            question=query
        )

        # Generate answer using local model
        start_time = time.time()

        results = self.pipe(
            prompt,
            num_return_sequences=1,
        )

        generation_time = time.time() - start_time
        print(f"Answer generated in {generation_time:.2f} seconds")

        answer = results[0]["generated_text"]
        return answer

    def answer_query_with_sources(
            self,
            query: str,
            documents: List[Tuple[Document, float]],
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Answer a query with source information.

        Args:
            query: User query
            documents: List of (document, score) tuples
            metadata_filter: Optional metadata filters used

        Returns:
            Tuple of (answer, sources)
        """
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
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        memory_info = {}

        # Get GPU memory usage if available
        if self.device.startswith("cuda") and torch.cuda.is_available():
            device_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
            memory_allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)

            memory_info.update({
                "memory_allocated_gb": f"{memory_allocated:.2f}",
                "memory_reserved_gb": f"{memory_reserved:.2f}",
            })

        # Model configuration info
        model_config = {
            "model_name": self.model_name,
            "device": self.device,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "quantization": "4-bit" if self.use_4bit else "8-bit" if self.use_8bit else "none",
            "torch_dtype": str(self.torch_dtype),
        }

        return {**model_config, **memory_info}