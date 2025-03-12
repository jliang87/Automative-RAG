#!/bin/bash
# Script to download models for the Automotive Specs RAG system

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

# Always use host models directory for downloading
HOST_MODELS_DIR=${HOST_MODELS_DIR:-"models"}

# Set paths for specific model types
EMBEDDING_DIR="${HOST_MODELS_DIR}/${EMBEDDING_MODEL_PATH:-embeddings}"
COLBERT_DIR="${HOST_MODELS_DIR}/${COLBERT_MODEL_PATH:-colbert}"
LLM_DIR="${HOST_MODELS_DIR}/${LLM_MODEL_PATH:-llm}"
WHISPER_DIR="${HOST_MODELS_DIR}/${WHISPER_MODEL_PATH:-whisper}"

# Get default model names from environment or use defaults
DEFAULT_EMBEDDING_MODEL_NAME=${DEFAULT_EMBEDDING_MODEL:-"bge-small-en-v1.5"}
DEFAULT_COLBERT_MODEL_NAME=${DEFAULT_COLBERT_MODEL:-"colbertv2.0"}
DEFAULT_LLM_MODEL_NAME=${DEFAULT_LLM_MODEL:-"DeepSeek-R1-Distill-Qwen-7B"}
DEFAULT_WHISPER_MODEL_NAME=${DEFAULT_WHISPER_MODEL:-"medium"}

# Model identifiers (Hugging Face)
BGE_MODEL_ID=${HF_EMBEDDING_MODEL:-"BAAI/bge-small-en-v1.5"}
COLBERT_MODEL_ID=${HF_COLBERT_MODEL:-"colbert-ir/colbertv2.0"}
DEEPSEEK_MODEL_ID=${HF_DEEPSEEK_MODEL:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}

# Set actual model directories with specific model names
EMBEDDING_MODEL_DIR="${EMBEDDING_DIR}/${DEFAULT_EMBEDDING_MODEL_NAME}"
COLBERT_MODEL_DIR="${COLBERT_DIR}/${DEFAULT_COLBERT_MODEL_NAME}"
LLM_MODEL_DIR="${LLM_DIR}/${DEFAULT_LLM_MODEL_NAME}"
WHISPER_MODEL_DIR="${WHISPER_DIR}/${DEFAULT_WHISPER_MODEL_NAME}"

# Create directories if they don't exist
mkdir -p "${EMBEDDING_MODEL_DIR}" "${COLBERT_MODEL_DIR}" "${LLM_MODEL_DIR}" "${WHISPER_MODEL_DIR}"

echo "Starting model downloads to host system (${HOST_MODELS_DIR})..."
echo "These models will be used by the container via volume mount to ${CONTAINER_MODELS_DIR}"

# Check for Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+ and try again."
    exit 1
fi

# Install required packages
echo "Installing required packages..."
pip install torch torchvision torchaudio sentence-transformers transformers bitsandbytes accelerate tqdm
pip install openai-whisper

echo "Starting model downloads..."

# Download BGE model for embeddings
echo "Downloading embedding model: ${BGE_MODEL_ID}..."
python -c "
from sentence_transformers import SentenceTransformer
import os
cache_dir = os.path.abspath('${EMBEDDING_MODEL_DIR}')
model = SentenceTransformer('${BGE_MODEL_ID}', cache_folder=cache_dir)
print('Embedding model downloaded successfully.')
"

if [ $? -ne 0 ]; then
    echo "Error downloading embedding model."
    exit 1
fi

# Download ColBERT model
echo "Downloading ColBERT model: ${COLBERT_MODEL_ID}..."
python -c "
from transformers import AutoTokenizer
import os
cache_dir = os.path.abspath('${COLBERT_MODEL_DIR}')
os.environ['HF_HOME'] = cache_dir  # Use HF_HOME instead of TRANSFORMERS_CACHE
tokenizer = AutoTokenizer.from_pretrained('${COLBERT_MODEL_ID}', use_fast=True, cache_dir=cache_dir)
print('ColBERT model downloaded successfully.')
"

if [ $? -ne 0 ]; then
    echo "Error downloading ColBERT model."
    exit 1
fi

# Download DeepSeek model
echo "Downloading DeepSeek model: ${DEEPSEEK_MODEL_ID}..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
cache_dir = os.path.abspath('${LLM_MODEL_DIR}')
os.environ['HF_HOME'] = cache_dir  # Use HF_HOME instead of TRANSFORMERS_CACHE
# Just download the tokenizer first (much smaller)
print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('${DEEPSEEK_MODEL_ID}', trust_remote_code=True, cache_dir=cache_dir)

# Ask for confirmation before downloading the full model
print('Tokenizer downloaded. The full model is several GB in size.')
"

read -p "Do you want to download the full DeepSeek model? This will use several GB of disk space. (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
cache_dir = os.path.abspath('${LLM_MODEL_DIR}')
os.environ['HF_HOME'] = cache_dir  # Use HF_HOME instead of TRANSFORMERS_CACHE
print('Downloading full model...')
tokenizer = AutoTokenizer.from_pretrained('${DEEPSEEK_MODEL_ID}', trust_remote_code=True, cache_dir=cache_dir)

# Configure quantization (reduces size and memory usage)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True
)

# Download the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    '${DEEPSEEK_MODEL_ID}',
    quantization_config=quantization_config,
    device_map='auto',
    trust_remote_code=True,
    cache_dir=cache_dir
)
print('DeepSeek model downloaded successfully.')
"
    if [ $? -ne 0 ]; then
        echo "Error downloading DeepSeek model."
        exit 1
    fi
else
    echo "Skipping full model download."
fi

# Download Whisper model
echo "Downloading Whisper ${DEFAULT_WHISPER_MODEL_NAME} model..."
python -c "
import whisper
import os
os.environ['XDG_CACHE_HOME'] = os.path.abspath('${WHISPER_MODEL_DIR}')
model = whisper.load_model('${DEFAULT_WHISPER_MODEL_NAME}')
print('Whisper model downloaded successfully.')
"

if [ $? -ne 0 ]; then
    echo "Error downloading Whisper model."
    exit 1
fi

echo "All models downloaded successfully!"
echo "Model locations on host system:"
echo "- Embedding model: ${EMBEDDING_MODEL_DIR}"
echo "- ColBERT model: ${COLBERT_MODEL_DIR}"
echo "- DeepSeek LLM: ${LLM_MODEL_DIR}"
echo "- Whisper model: ${WHISPER_MODEL_DIR}"
echo ""
echo "These models will be available to containers at:"
echo "- Embedding model: ${CONTAINER_MODELS_DIR}/${EMBEDDING_MODEL_PATH}/${DEFAULT_EMBEDDING_MODEL_NAME}"
echo "- ColBERT model: ${CONTAINER_MODELS_DIR}/${COLBERT_MODEL_PATH}/${DEFAULT_COLBERT_MODEL_NAME}"
echo "- DeepSeek LLM: ${CONTAINER_MODELS_DIR}/${LLM_MODEL_PATH}/${DEFAULT_LLM_MODEL_NAME}"
echo "- Whisper model: ${CONTAINER_MODELS_DIR}/${WHISPER_MODEL_PATH}/${DEFAULT_WHISPER_MODEL_NAME}"