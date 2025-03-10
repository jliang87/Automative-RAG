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

echo "Starting model downloads..."

# Check for Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+ and try again."
    exit 1
fi

# Download BGE model for embeddings
echo "Downloading embedding model: $BGE_MODEL..."
python -c "
from sentence_transformers import SentenceTransformer
import os
os.environ['TRANSFORMERS_CACHE'] = os.path.abspath('$EMBEDDING_DIR')
model = SentenceTransformer('$BGE_MODEL')
print('Embedding model downloaded successfully.')
"

if [ $? -ne 0 ]; then
    echo "Error downloading embedding model."
    exit 1
fi

# Download ColBERT model
echo "Downloading ColBERT model: $COLBERT_MODEL..."
python -c "
from transformers import AutoTokenizer
import os
os.environ['TRANSFORMERS_CACHE'] = os.path.abspath('$COLBERT_DIR')
tokenizer = AutoTokenizer.from_pretrained('$COLBERT_MODEL', use_fast=True)
print('ColBERT model downloaded successfully.')
"

if [ $? -ne 0 ]; then
    echo "Error downloading ColBERT model."
    exit 1
fi

# Download DeepSeek model
echo "Downloading DeepSeek model: $DEEPSEEK_MODEL..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
os.environ['TRANSFORMERS_CACHE'] = os.path.abspath('$LLM_DIR')

# Just download the tokenizer first (much smaller)
print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('$DEEPSEEK_MODEL', trust_remote_code=True)

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
os.environ['TRANSFORMERS_CACHE'] = os.path.abspath('$LLM_DIR')

print('Downloading full model...')
tokenizer = AutoTokenizer.from_pretrained('$DEEPSEEK_MODEL', trust_remote_code=True)

# Configure quantization (reduces size and memory usage)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True
)

# Download the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    '$DEEPSEEK_MODEL',
    quantization_config=quantization_config,
    device_map='auto',
    trust_remote_code=True
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
echo "Downloading Whisper $WHISPER_MODEL model..."
python -c "
import whisper
import os
os.environ['XDG_CACHE_HOME'] = os.path.abspath('$WHISPER_DIR')
model = whisper.load_model('$WHISPER_MODEL')
print('Whisper model downloaded successfully.')
"

if [ $? -ne 0 ]; then
    echo "Error downloading Whisper model."
    exit 1
fi

echo "All models downloaded successfully!"
echo "Model locations:"
echo "- Embedding model: $EMBEDDING_DIR"
echo "- ColBERT model: $COLBERT_DIR"
echo "- DeepSeek LLM: $LLM_DIR"
echo "- Whisper model: $WHISPER_DIR"
echo ""
echo "You can update model paths in the .env file if needed."