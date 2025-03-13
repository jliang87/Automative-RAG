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
WHISPER_MODEL_ID=${HF_WHISPER_MODEL:-"openai/whisper-medium"}

# Set actual model directories with specific model names
EMBEDDING_MODEL_DIR="${EMBEDDING_DIR}/${DEFAULT_EMBEDDING_MODEL_NAME}"
COLBERT_MODEL_DIR="${COLBERT_DIR}/${DEFAULT_COLBERT_MODEL_NAME}"
LLM_MODEL_DIR="${LLM_DIR}/${DEFAULT_LLM_MODEL_NAME}"
WHISPER_MODEL_DIR="${WHISPER_DIR}/${DEFAULT_WHISPER_MODEL_NAME}"

# Create directories if they don't exist
mkdir -p "${EMBEDDING_MODEL_DIR}" "${COLBERT_MODEL_DIR}" "${LLM_MODEL_DIR}" "${WHISPER_MODEL_DIR}"

echo "Starting model downloads to host system (${HOST_MODELS_DIR})..."
echo "These models will be used by the container via volume mount to ${CONTAINER_MODELS_DIR}"

echo "Starting model downloads..."

# Function to check if a directory has content
download_if_empty() {
    MODEL_ID=$1
    LOCAL_DIR=$2
    MODEL_NAME=$3

    if [ -d "$LOCAL_DIR" ] && [ "$(ls -A "$LOCAL_DIR")" ]; then
        echo "$MODEL_NAME already exists in $LOCAL_DIR, skipping download."
    else
        echo "Downloading $MODEL_NAME..."
        hfd.sh "$MODEL_ID" --local-dir "$LOCAL_DIR"
        echo "$MODEL_NAME downloaded successfully."
    fi
}

# Download models only if their respective directories are empty
download_if_empty "$BGE_MODEL_ID" "$EMBEDDING_MODEL_DIR" "Embedding model"
download_if_empty "$COLBERT_MODEL_ID" "$COLBERT_MODEL_DIR" "ColBERT model"
download_if_empty "$DEEPSEEK_MODEL_ID" "$LLM_MODEL_DIR" "DeepSeek model"
download_if_empty "$WHISPER_MODEL_ID" "$WHISPER_MODEL_DIR" "Whisper model"

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