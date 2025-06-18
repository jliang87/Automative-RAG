from typing import Dict, Any, Optional
from enum import Enum
import logging
from src.config.settings import settings

logger = logging.getLogger(__name__)


class QueryMode(str, Enum):
    """Query modes with specific configuration needs."""
    FACTS = "facts"
    FEATURES = "features"
    TRADEOFFS = "tradeoffs"
    SCENARIOS = "scenarios"
    DEBATE = "debate"
    QUOTES = "quotes"


class ModeSpecificConfig:
    """
    Mode-specific configuration for LLM parameters, document filtering, and token management.

    This centralizes all mode-specific behaviors that were previously hardcoded.
    """

    def __init__(self):
        # Mode-specific LLM parameters
        self.mode_configs = {
            QueryMode.FACTS: {
                # Conservative settings for factual accuracy
                "temperature": 0.0,
                "max_tokens": 400,  # Shorter, focused answers
                "top_p": 0.8,
                "repetition_penalty": 1.15,

                # Document filtering
                "retrieval_k": 20,
                "final_k": 8,
                "relevance_cutoff": 0.3,  # Stricter cutoff for facts
                "confidence_cutoff": 0.7,

                # Token management
                "max_context_tokens": 2000,
                "docs_per_source": 2,
                "prioritize_numerical": True,
            },

            QueryMode.FEATURES: {
                # Moderate creativity for analysis
                "temperature": 0.1,
                "max_tokens": 600,  # More space for analysis
                "top_p": 0.85,
                "repetition_penalty": 1.1,

                # Document filtering
                "retrieval_k": 30,
                "final_k": 12,
                "relevance_cutoff": 0.25,  # More lenient for broader context
                "confidence_cutoff": 0.6,

                # Token management
                "max_context_tokens": 3000,
                "docs_per_source": 3,
                "prioritize_numerical": False,
            },

            QueryMode.TRADEOFFS: {
                # Balanced creativity for pros/cons analysis
                "temperature": 0.15,
                "max_tokens": 700,
                "top_p": 0.9,
                "repetition_penalty": 1.1,

                # Document filtering
                "retrieval_k": 35,
                "final_k": 15,
                "relevance_cutoff": 0.2,  # Most lenient for diverse perspectives
                "confidence_cutoff": 0.5,

                # Token management
                "max_context_tokens": 3500,
                "docs_per_source": 3,
                "prioritize_numerical": False,
            },

            QueryMode.SCENARIOS: {
                # Moderate creativity for scenario analysis
                "temperature": 0.12,
                "max_tokens": 650,
                "top_p": 0.87,
                "repetition_penalty": 1.1,

                # Document filtering
                "retrieval_k": 30,
                "final_k": 12,
                "relevance_cutoff": 0.25,
                "confidence_cutoff": 0.6,

                # Token management
                "max_context_tokens": 3200,
                "docs_per_source": 3,
                "prioritize_numerical": False,
            },

            QueryMode.DEBATE: {
                # Higher creativity for multiple perspectives
                "temperature": 0.2,
                "max_tokens": 800,  # Most space for debate format
                "top_p": 0.92,
                "repetition_penalty": 1.05,

                # Document filtering
                "retrieval_k": 40,
                "final_k": 18,
                "relevance_cutoff": 0.2,  # Very lenient for diverse viewpoints
                "confidence_cutoff": 0.5,

                # Token management
                "max_context_tokens": 4000,
                "docs_per_source": 4,  # More diversity needed
                "prioritize_numerical": False,
            },

            QueryMode.QUOTES: {
                # Conservative for exact quote extraction
                "temperature": 0.05,
                "max_tokens": 500,
                "top_p": 0.75,
                "repetition_penalty": 1.2,  # Avoid repetitive quotes

                # Document filtering
                "retrieval_k": 25,
                "final_k": 10,
                "relevance_cutoff": 0.3,  # Need relevant sources for quotes
                "confidence_cutoff": 0.65,

                # Token management
                "max_context_tokens": 2500,
                "docs_per_source": 2,
                "prioritize_numerical": False,
            }
        }

    def get_llm_params(self, mode: QueryMode) -> Dict[str, Any]:
        """Get LLM generation parameters for a specific mode."""
        config = self.mode_configs.get(mode, self.mode_configs[QueryMode.FACTS])

        return {
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            "top_p": config.get("top_p", 0.85),
            "repetition_penalty": config.get("repetition_penalty", 1.1),
        }

    def get_retrieval_params(self, mode: QueryMode) -> Dict[str, Any]:
        """Get document retrieval parameters for a specific mode."""
        config = self.mode_configs.get(mode, self.mode_configs[QueryMode.FACTS])

        return {
            "retrieval_k": config["retrieval_k"],
            "final_k": config["final_k"],
            "relevance_cutoff": config["relevance_cutoff"],
            "confidence_cutoff": config["confidence_cutoff"],
        }

    def get_context_params(self, mode: QueryMode) -> Dict[str, Any]:
        """Get context/token management parameters for a specific mode."""
        config = self.mode_configs.get(mode, self.mode_configs[QueryMode.FACTS])

        return {
            "max_context_tokens": config["max_context_tokens"],
            "docs_per_source": config["docs_per_source"],
            "prioritize_numerical": config["prioritize_numerical"],
        }

    def should_trim_low_relevance(self, mode: QueryMode, relevance_score: float) -> bool:
        """Check if a document should be trimmed based on relevance score."""
        config = self.mode_configs.get(mode, self.mode_configs[QueryMode.FACTS])
        return relevance_score < config["relevance_cutoff"]

    def should_trim_low_confidence(self, mode: QueryMode, confidence_score: float) -> bool:
        """Check if a document should be trimmed based on confidence score."""
        config = self.mode_configs.get(mode, self.mode_configs[QueryMode.FACTS])
        return confidence_score < config["confidence_cutoff"]

    def get_mode_complexity(self, mode: QueryMode) -> str:
        """Get complexity level for mode (for estimation and logging)."""
        complexity_map = {
            QueryMode.FACTS: "simple",
            QueryMode.FEATURES: "moderate",
            QueryMode.TRADEOFFS: "complex",
            QueryMode.SCENARIOS: "complex",
            QueryMode.DEBATE: "complex",
            QueryMode.QUOTES: "simple"
        }
        return complexity_map.get(mode, "moderate")


# Global instance
mode_config = ModeSpecificConfig()


def estimate_token_count(text: str) -> int:
    """
    Rough token count estimation for Chinese/English mixed text.
    Chinese characters ≈ 1.5 tokens, English words ≈ 1.3 tokens
    """
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_chars = len(text) - chinese_chars

    # Rough estimation
    chinese_tokens = chinese_chars * 1.5
    english_tokens = (english_chars / 4) * 1.3  # Assume avg 4 chars per word

    return int(chinese_tokens + english_tokens)


def trim_documents_by_tokens(documents, mode: QueryMode, max_tokens: Optional[int] = None) -> list:
    """
    Trim documents to fit within token limits while preserving quality.

    Prioritization order:
    1. Higher relevance scores
    2. Numerical data (for facts mode)
    3. Diversity (different sources)
    """
    if not documents:
        return documents

    context_params = mode_config.get_context_params(mode)
    target_tokens = max_tokens or context_params["max_context_tokens"]

    # Sort by relevance score (assuming (doc, score) tuples)
    if isinstance(documents[0], tuple):
        sorted_docs = sorted(documents, key=lambda x: x[1], reverse=True)
    else:
        # Fallback if no scores
        sorted_docs = [(doc, 1.0) for doc in documents]

    # Track tokens and source diversity
    total_tokens = 0
    selected_docs = []
    source_counts = {}
    max_per_source = context_params["docs_per_source"]

    for doc, score in sorted_docs:
        # Apply relevance cutoff
        if mode_config.should_trim_low_relevance(mode, score):
            continue

        # Check source diversity
        if isinstance(doc, tuple):
            content = doc[0].page_content if hasattr(doc[0], 'page_content') else str(doc[0])
            metadata = doc[0].metadata if hasattr(doc[0], 'metadata') else {}
        else:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}

        source_id = metadata.get('source_id', 'unknown')

        if source_counts.get(source_id, 0) >= max_per_source:
            continue

        # Estimate token count
        doc_tokens = estimate_token_count(content)

        # Check if adding this doc would exceed limit
        if total_tokens + doc_tokens > target_tokens and selected_docs:
            break

        # Add the document
        selected_docs.append((doc, score))
        total_tokens += doc_tokens
        source_counts[source_id] = source_counts.get(source_id, 0) + 1

        logger.debug(f"Added doc from {source_id}, total tokens: {total_tokens}")

    logger.info(f"Trimmed documents for {mode}: {len(documents)} -> {len(selected_docs)} docs, ~{total_tokens} tokens")

    return selected_docs