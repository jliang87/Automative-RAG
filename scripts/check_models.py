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

# Import settings from our app
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from src.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


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
                model_path = os.path.join(settings.models_dir, "embeddings", model_id)

            logger.info(f"Downloading embedding model to {model_path}")
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


def check_whisper_model(model_path, model_size):
    """Check if Whisper model exists and load it"""
    logger.info(f"Checking Whisper model: {model_path}")

    try:
        # We don't actually try to load the model here as it can be large
        # Just check if the directory exists and has files
        if os.path.exists(model_path) and len(os.listdir(model_path)) > 0:
            logger.info(f"Whisper model found at {model_path}")
            return True
        else:
            logger.info(f"Whisper model not found at {model_path}")
            logger.info(f"Whisper models should be downloaded using download_models.sh script")
            return False
    except Exception as e:
        logger.error(f"Error checking Whisper model: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check if model files exist and download if necessary")
    parser.add_argument("--check-only", action="store_true", help="Only check if models exist, don't download")
    args = parser.parse_args()

    logger.info("Checking model availability...")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA current device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    # Use the settings to get model paths
    logger.info(f"Running in {'container' if settings.is_container else 'host'} environment")
    logger.info(f"Models base directory: {settings.models_dir}")
    logger.info(f"Checking models at the following paths:")
    logger.info(f"- Embedding model: {settings.embedding_model_full_path}")
    logger.info(f"- ColBERT model: {settings.colbert_model_full_path}")
    logger.info(f"- LLM model: {settings.llm_model_full_path}")
    logger.info(f"- Whisper model: {settings.whisper_model_full_path}")

    # Check and load each model
    embedding_ok = check_embedding_model(settings.embedding_model_full_path)
    colbert_ok = check_colbert_model(settings.colbert_model_full_path)
    llm_ok = check_deepseek_model(settings.llm_model_full_path)
    whisper_ok = check_whisper_model(settings.whisper_model_full_path, settings.whisper_model_size)

    # Check if all models are available
    all_models_ok = embedding_ok and colbert_ok and llm_ok and whisper_ok

    if all_models_ok:
        logger.info("All models are available and ready to use")
        sys.exit(0)
    else:
        logger.error("Some models are missing or invalid")
        if not embedding_ok:
            logger.error("Embedding model is missing or invalid")
        if not colbert_ok:
            logger.error("ColBERT model is missing or invalid")
        if not llm_ok:
            logger.error("LLM model is missing or invalid")
        if not whisper_ok:
            logger.error("Whisper model is missing or invalid")

        logger.error("Please run the download_models.sh script on the host system")
        sys.exit(1)


if __name__ == "__main__":
    main()