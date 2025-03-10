#!/bin/bash
# Script to download models for the Automotive Specs RAG system
# Modified version for regions where Hugging Face is not accessible (e.g., China)

# First, check if .env file exists, if not create from .env.example
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Created .env file from .env.example"
    else
        echo "Error: .env.example not found"
        exit 1
    fi
fi

# Load environment variables from .env file
set -a
source .env
set +a

# Check if running in Docker or locally
IN_DOCKER=false
if [ -f "/.dockerenv" ]; then
    IN_DOCKER=true
    echo "Running in Docker environment"
else
    echo "Running in local environment"
fi

# Set default model directories if not specified in .env
# Use appropriate paths based on environment (Docker vs local)
if [ "$IN_DOCKER" = true ]; then
    # Docker paths (absolute)
    DEFAULT_EMBEDDING_DIR="/app/models/embeddings"
    DEFAULT_COLBERT_DIR="/app/models/colbert"
    DEFAULT_LLM_DIR="/app/models/llm"
    DEFAULT_WHISPER_DIR="/app/models/whisper"
else
    # Local paths (relative)
    DEFAULT_EMBEDDING_DIR="models/embeddings"
    DEFAULT_COLBERT_DIR="models/colbert"
    DEFAULT_LLM_DIR="models/llm"
    DEFAULT_WHISPER_DIR="models/whisper"
fi

EMBEDDING_DIR=${EMBEDDING_CACHE_DIR:-$DEFAULT_EMBEDDING_DIR}
COLBERT_DIR=${COLBERT_MODEL:-$DEFAULT_COLBERT_DIR}
LLM_DIR=${LLM_CACHE_DIR:-$DEFAULT_LLM_DIR}
WHISPER_DIR=${WHISPER_CACHE_DIR:-$DEFAULT_WHISPER_DIR}

# Model identifiers
BGE_MODEL="BAAI/bge-small-en-v1.5"
COLBERT_MODEL="colbert-ir/colbertv2.0"
DEEPSEEK_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
WHISPER_MODEL=${WHISPER_MODEL_SIZE:-"medium"}

# Create directories if they don't exist
mkdir -p "$EMBEDDING_DIR" "$COLBERT_DIR" "$LLM_DIR" "$WHISPER_DIR"

echo "Starting model downloads using China mirrors..."

# Check for Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+ and try again."
    exit 1
fi

# Install required packages with mirror
echo "Installing required packages from mirrors..."
pip install -i https://mirror.baidu.com/pypi/simple/ transformers torch sentence-transformers tqdm bitsandbytes accelerate

# Whisper might need a different mirror
echo "Installing openai-whisper from mirrors..."
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ openai-whisper

# Configure model mirrors in .env or script
HF_MIRROR="https://hf-mirror.com"
MODELSCOPE_MIRROR="https://modelscope.cn/models"

# Configure Python to use mirrors
cat > mirror_download.py << EOL
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer
import whisper
import json
from pathlib import Path
import requests
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

# Use Hugging Face mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Use modelscope as alternative
MODELSCOPE_ENDPOINT = "https://modelscope.cn/api/v1/models"

def download_file(url, local_path):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    # Create parent directories if they don't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    logger.info(f"Downloading {url} to {local_path}")
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))

    progress_bar.close()
    return local_path

def download_embedding_model(model_name, output_dir):
    """Download embedding model using mirrors."""
    logger.info(f"Downloading embedding model: {model_name}")
    try:
        # Try to use modelscope first
        model_id = model_name.split('/')[-1]
        logger.info(f"First trying sentence-transformers from mirror...")
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = output_dir
        # Use SentenceTransformer with mirror
        model = SentenceTransformer(model_name)
        model.save(output_dir)
        logger.info(f"Successfully downloaded embedding model to {output_dir}")
        return True
    except Exception as e:
        logger.warning(f"Error with SentenceTransformer approach: {str(e)}")
        logger.info("Trying backup method...")

        try:
            # Backup method: manually download key files
            model_files = [
                "config.json",
                "pytorch_model.bin",
                "sentence_bert_config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "vocab.txt"
            ]

            base_url = f"{os.environ.get('HF_ENDPOINT')}/{model_name}/resolve/main"
            for file in model_files:
                file_url = f"{base_url}/{file}"
                output_path = os.path.join(output_dir, file)
                try:
                    download_file(file_url, output_path)
                except Exception as file_error:
                    logger.warning(f"Failed to download {file}: {str(file_error)}")

            logger.info(f"Downloaded embedding model files to {output_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to download embedding model: {str(e)}")
            return False

def download_colbert_model(model_name, output_dir):
    """Download ColBERT model using mirrors."""
    logger.info(f"Downloading ColBERT model: {model_name}")
    try:
        # For ColBERT, just download the tokenizer files
        tokenizer_files = [
            "config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.txt"
        ]

        base_url = f"{os.environ.get('HF_ENDPOINT')}/{model_name}/resolve/main"
        for file in tokenizer_files:
            file_url = f"{base_url}/{file}"
            output_path = os.path.join(output_dir, file)
            try:
                download_file(file_url, output_path)
            except Exception as file_error:
                logger.warning(f"Failed to download {file}: {str(file_error)}")

        logger.info(f"Downloaded ColBERT model files to {output_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to download ColBERT model: {str(e)}")
        return False

def download_deepseek_model(model_name, output_dir):
    """Download DeepSeek model using mirrors."""
    logger.info(f"Downloading DeepSeek model: {model_name}")
    try:
        # First just download the tokenizer files
        tokenizer_files = [
            "config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model"
        ]

        base_url = f"{os.environ.get('HF_ENDPOINT')}/{model_name}/resolve/main"
        for file in tokenizer_files:
            file_url = f"{base_url}/{file}"
            output_path = os.path.join(output_dir, file)
            try:
                download_file(file_url, output_path)
            except Exception as file_error:
                logger.warning(f"Failed to download {file}: {str(file_error)}")

        logger.info(f"Downloaded DeepSeek tokenizer files to {output_dir}")

        # Ask for confirmation before downloading the full model
        download_full = input("Do you want to download the full DeepSeek model? This will use several GB of disk space. (y/n): ")

        if download_full.lower() == 'y':
            logger.info("Downloading full model (this will take some time)...")

            # Model files to download (these are the essential ones)
            model_files = [
                "pytorch_model-00001-of-00003.bin",
                "pytorch_model-00002-of-00003.bin",
                "pytorch_model-00003-of-00003.bin",
                "pytorch_model.bin.index.json"
            ]

            for file in model_files:
                file_url = f"{base_url}/{file}"
                output_path = os.path.join(output_dir, file)
                try:
                    download_file(file_url, output_path)
                except Exception as file_error:
                    logger.error(f"Failed to download {file}: {str(file_error)}")
                    return False

            logger.info(f"Downloaded DeepSeek full model to {output_dir}")
        else:
            logger.info("Skipping full model download.")

        return True
    except Exception as e:
        logger.error(f"Failed to download DeepSeek model: {str(e)}")
        return False

def download_whisper_model(model_size, output_dir):
    """Download Whisper model using China mirrors."""
    logger.info(f"Downloading Whisper {model_size} model...")

    # Define alternative mirrors for Whisper models
    # ModelScope mirror for Whisper models
    modelscope_whisper_models = {
        "tiny": "https://modelscope.cn/api/v1/models/AI-ModelScope/whisper-tiny/repo?Revision=master&FilePath=model.bin",
        "base": "https://modelscope.cn/api/v1/models/AI-ModelScope/whisper-base/repo?Revision=master&FilePath=model.bin",
        "small": "https://modelscope.cn/api/v1/models/AI-ModelScope/whisper-small/repo?Revision=master&FilePath=model.bin",
        "medium": "https://modelscope.cn/api/v1/models/AI-ModelScope/whisper-medium/repo?Revision=master&FilePath=model.bin",
        "large": "https://modelscope.cn/api/v1/models/AI-ModelScope/whisper-large/repo?Revision=master&FilePath=model.bin"
    }

    # HuggingFace mirror alternative URLs (backup)
    hf_mirror_whisper_models = {
        "tiny": "https://hf-mirror.com/openai/whisper-tiny/resolve/main/pytorch_model.bin",
        "base": "https://hf-mirror.com/openai/whisper-base/resolve/main/pytorch_model.bin",
        "small": "https://hf-mirror.com/openai/whisper-small/resolve/main/pytorch_model.bin",
        "medium": "https://hf-mirror.com/openai/whisper-medium/resolve/main/pytorch_model.bin",
        "large": "https://hf-mirror.com/openai/whisper-large-v2/resolve/main/pytorch_model.bin"
    }

    # Wisemodel mirror (another alternative)
    wisemodel_whisper_models = {
        "tiny": "https://wisemodel.cn/models/openai/whisper-tiny/pytorch_model.bin",
        "base": "https://wisemodel.cn/models/openai/whisper-base/pytorch_model.bin",
        "small": "https://wisemodel.cn/models/openai/whisper-small/pytorch_model.bin",
        "medium": "https://wisemodel.cn/models/openai/whisper-medium/pytorch_model.bin",
        "large": "https://wisemodel.cn/models/openai/whisper-large-v2/pytorch_model.bin"
    }

    try:
        # Create output directory structure
        whisper_model_dir = os.path.join(output_dir, model_size)
        os.makedirs(whisper_model_dir, exist_ok=True)

        # First try the standard method in case it works
        try:
            logger.info("Trying standard whisper download method first...")
            os.environ["XDG_CACHE_HOME"] = output_dir
            model = whisper.load_model(model_size)
            logger.info(f"Successfully downloaded Whisper model using standard method")
            return True
        except Exception as e:
            logger.warning(f"Standard method failed: {str(e)}")
            logger.info("Trying alternative mirrors...")

        # Try ModelScope mirror
        try:
            if model_size in modelscope_whisper_models:
                model_url = modelscope_whisper_models[model_size]
                output_path = os.path.join(whisper_model_dir, "model.bin")
                logger.info(f"Downloading from ModelScope mirror...")
                download_file(model_url, output_path)
                logger.info(f"Successfully downloaded Whisper model from ModelScope")
                return True
        except Exception as e:
            logger.warning(f"ModelScope mirror failed: {str(e)}")

        # Try HuggingFace mirror
        try:
            if model_size in hf_mirror_whisper_models:
                model_url = hf_mirror_whisper_models[model_size]
                output_path = os.path.join(whisper_model_dir, "pytorch_model.bin")
                logger.info(f"Downloading from HF mirror...")
                download_file(model_url, output_path)

                # Also download config file
                config_url = f"https://hf-mirror.com/openai/whisper-{model_size}/resolve/main/config.json"
                config_path = os.path.join(whisper_model_dir, "config.json")
                download_file(config_url, config_path)

                logger.info(f"Successfully downloaded Whisper model from HF mirror")
                return True
        except Exception as e:
            logger.warning(f"HF mirror failed: {str(e)}")

        # Try Wisemodel mirror as last resort
        try:
            if model_size in wisemodel_whisper_models:
                model_url = wisemodel_whisper_models[model_size]
                output_path = os.path.join(whisper_model_dir, "pytorch_model.bin")
                logger.info(f"Downloading from Wisemodel mirror...")
                download_file(model_url, output_path)

                # Create a basic config file if needed
                config_path = os.path.join(whisper_model_dir, "config.json")
                if not os.path.exists(config_path):
                    with open(config_path, 'w') as f:
                        json.dump({"model_type": "whisper"}, f)

                logger.info(f"Successfully downloaded Whisper model from Wisemodel mirror")
                return True
        except Exception as e:
            logger.warning(f"Wisemodel mirror failed: {str(e)}")

        logger.error("All mirrors failed. Could not download Whisper model.")
        return False
    except Exception as e:
        logger.error(f"Failed to download Whisper model: {str(e)}")
        return False

if __name__ == "__main__":
    import sys

    model_type = sys.argv[1] if len(sys.argv) > 1 else "all"

    if model_type == "embedding" or model_type == "all":
        download_embedding_model("${BGE_MODEL}", "${EMBEDDING_DIR}")

    if model_type == "colbert" or model_type == "all":
        download_colbert_model("${COLBERT_MODEL}", "${COLBERT_DIR}")

    if model_type == "deepseek" or model_type == "all":
        download_deepseek_model("${DEEPSEEK_MODEL}", "${LLM_DIR}")

    if model_type == "whisper" or model_type == "all":
        download_whisper_model("${WHISPER_MODEL}", "${WHISPER_DIR}")
EOL

# Download embedding model
echo "Downloading embedding model..."
python mirror_download.py embedding

# Download ColBERT model
echo "Downloading ColBERT model..."
python mirror_download.py colbert

# Download DeepSeek model
echo "Downloading DeepSeek model..."
python mirror_download.py deepseek

# Download Whisper model
echo "Downloading Whisper model..."
python mirror_download.py whisper

echo "All models downloaded successfully (or skipped)!"
echo "Model locations:"
echo "- Embedding model: $EMBEDDING_DIR"
echo "- ColBERT model: $COLBERT_DIR"
echo "- DeepSeek LLM: $LLM_DIR"
echo "- Whisper model: $WHISPER_DIR"
echo ""
echo "You can update model paths in the .env file if needed."

# Cleanup temp file
rm -f mirror_download.py