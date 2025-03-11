#!/usr/bin/env python3
"""
Check if model files exist at specified paths, and download them if necessary.
This script is run during container startup to ensure all required models are available.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import argparse
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def load_env_variables():
    """Load environment variables from .env file"""
    # Try to load from .env file
    load_dotenv()

    # Get model paths from environment variables
    # Get default model names
    default_embedding_model = os.getenv("DEFAULT_EMBEDDING_MODEL", "bge-small-en-v1.5")
    default_colbert_model = os.getenv("DEFAULT_COLBERT_MODEL", "colbertv2.0")
    default_llm_model = os.getenv("DEFAULT_LLM_MODEL", "DeepSeek-R1-Distill-Qwen-7B")

    # For embedding model
    embedding_base_dir = os.getenv("EMBEDDING_MODEL", "/app/models/embeddings")
    embedding_model = os.path.join(embedding_base_dir, default_embedding_model)

    # For ColBERT model
    colbert_base_dir = os.getenv("COLBERT_MODEL", "/app/models/colbert")
    colbert_model = os.path.join(colbert_base_dir, default_colbert_model)

    # For DeepSeek model
    deepseek_base_dir = os.getenv("DEEPSEEK_MODEL", "/app/models/llm")
    deepseek_model = os.path.join(deepseek_base_dir, default_llm_model)

    return embedding_model, colbert_model, deepseek_model


def check_embedding_model(model_path):
    """Check if embedding model exists and load or download it"""
    logger.info(f"Checking embedding model: {model_path}")

    try:
        if os.path.exists(model_path):
            logger.info(f"Embedding model found at {model_path}")
            # Try to load the model to verify it's valid
            SentenceTransformer(model_path)
            return True
        else:
            logger.info(f"Embedding model not found at {model_path}, checking if it's a HuggingFace model ID")
            # Try to download from HuggingFace
            original_model_path = model_path
            if "/" in model_path:
                model_id = model_path.split("/")[-1]
                model_path = os.path.join(os.environ.get("TRANSFORMERS_CACHE", "/app/models/cache"), model_id)

            logger.info(f"Downloading embedding model {original_model_path} to {model_path}")
            model = SentenceTransformer(original_model_path)
            logger.info(f"Successfully loaded embedding model")
            return True
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        return False


def check_colbert_model(model_path):
    """Check if ColBERT model exists and load or download it"""
    logger.info(f"Checking ColBERT model: {model_path}")

    try:
        if os.path.exists(model_path):
            logger.info(f"ColBERT model found at {model_path}")
            # For ColBERT, just check if the directory exists
            return True
        else:
            logger.info(f"ColBERT model not found at {model_path}, checking if it's a HuggingFace model ID")
            # Try to initialize the tokenizer, which will download the model
            if "/" in model_path:
                colbert_tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                logger.info(f"Successfully initialized ColBERT tokenizer")
                return True
            else:
                logger.warning(f"ColBERT model path is invalid: {model_path}")
                return False
    except Exception as e:
        logger.error(f"Error loading ColBERT model: {str(e)}")
        return False


def check_deepseek_model(model_path):
    """Check if DeepSeek model exists and load or download it"""
    logger.info(f"Checking DeepSeek model: {model_path}")

    try:
        if os.path.exists(model_path):
            logger.info(f"DeepSeek model found at {model_path}")
            # Try to load the tokenizer to verify it's valid
            AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            return True
        else:
            logger.info(f"DeepSeek model not found at {model_path}, checking if it's a HuggingFace model ID")
            # Try to initialize the tokenizer, which will download the model
            if "/" in model_path:
                logger.info(f"Downloading DeepSeek tokenizer from {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                logger.info(f"Successfully initialized DeepSeek tokenizer")

                # For the LLM, just load the tokenizer to verify download, don't load the full model here
                return True
            else:
                logger.warning(f"DeepSeek model path is invalid: {model_path}")
                return False
    except Exception as e:
        logger.error(f"Error loading DeepSeek model: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check if model files exist and download if necessary")
    parser.add_argument("--check-only", action="store_true", help="Only check if models exist, don't download")
    args = parser.parse_args()

    # Load environment variables
    embedding_model, colbert_model, deepseek_model = load_env_variables()

    logger.info("Checking model availability...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # Check and load each model
    embedding_ok = check_embedding_model(embedding_model)
    colbert_ok = check_colbert_model(colbert_model)
    deepseek_ok = check_deepseek_model(deepseek_model)

    # Check if all models are available
    all_models_ok = embedding_ok and colbert_ok and deepseek_ok

    if all_models_ok:
        logger.info("All models are available and ready to use")
        sys.exit(0)
    else:
        logger.error("Some models are missing or invalid")
        sys.exit(1)


if __name__ == "__main__":
    main()