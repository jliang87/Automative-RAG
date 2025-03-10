#!/bin/bash
# download_models.sh

# Set model directories
EMBEDDING_DIR="models/embeddings"
COLBERT_DIR="models/colbert"
LLM_DIR="models/llm"
WHISPER_DIR="models/whisper"

# Create directories if they don't exist
mkdir -p "$EMBEDDING_DIR"
mkdir -p "$COLBERT_DIR"
mkdir -p "$LLM_DIR"
mkdir -p "$WHISPER_DIR"

# Function to download a model
download_model() {
    MODEL_NAME=$1
    OUTPUT_DIR=$2
    MODEL_TYPE=$3

    echo "======================================================"
    echo "Downloading $MODEL_TYPE model: $MODEL_NAME"
    echo "Output directory: $OUTPUT_DIR"
    echo "======================================================"

    # Run the download command with Poetry
    poetry run python -c "
import os
import sys
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

model_name = '$MODEL_NAME'
output_dir = '$OUTPUT_DIR'
model_type = '$MODEL_TYPE'

try:
    print(f'Downloading {model_type} model: {model_name}')

    # Download tokenizer first
    print('Downloading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    print(f'Tokenizer saved to {output_dir}')

    # Then download the model
    print('Downloading model (this may take some time)...')
    if model_type == 'llm':
        # For LLMs, use AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    else:
        # For embedding and other models
        model = AutoModel.from_pretrained(model_name)

    model.save_pretrained(output_dir)
    print(f'Model saved to {output_dir}')

    print(f'Successfully downloaded {model_type} model: {model_name}')
except Exception as e:
    print(f'Error downloading model: {str(e)}')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        echo "Download successful!"
    else
        echo "Download failed! Check the error message above."
    fi

    echo ""
}

# Function to download Whisper model
download_whisper() {
    MODEL_SIZE=$1
    OUTPUT_DIR=$2

    echo "======================================================"
    echo "Downloading Whisper model: $MODEL_SIZE"
    echo "Output directory: $OUTPUT_DIR"
    echo "======================================================"

    # Run the download command with Poetry
    poetry run python -c "
import os
import sys
import whisper

model_size = '$MODEL_SIZE'
output_dir = '$OUTPUT_DIR'

try:
    print(f'Downloading Whisper model: {model_size}')
    os.environ['WHISPER_MODELS_DIR'] = output_dir

    # This will download the model to the specified directory
    model = whisper.load_model(model_size)

    print(f'Successfully downloaded Whisper model: {model_size}')
    print(f'Model saved to {output_dir}')
except Exception as e:
    print(f'Error downloading Whisper model: {str(e)}')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        echo "Whisper download successful!"
    else
        echo "Whisper download failed! Check the error message above."
    fi

    echo ""
}

# Ask user which models to download
echo "Which models would you like to download?"
echo "1) All models (recommended)"
echo "2) Embedding model only (BGE)"
echo "3) ColBERT model only"
echo "4) LLM only (DeepSeek)"
echo "5) Whisper only"
echo "6) Exit"

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        # Download all models
        download_model "BAAI/bge-small-en-v1.5" "$EMBEDDING_DIR" "embedding"
        download_model "colbert-ir/colbertv2.0" "$COLBERT_DIR" "colbert"
        download_model "deepseek-ai/deepseek-coder-6.7b-instruct" "$LLM_DIR" "llm"
        download_whisper "medium" "$WHISPER_DIR"
        ;;
    2)
        # Download embedding model only
        download_model "BAAI/bge-small-en-v1.5" "$EMBEDDING_DIR" "embedding"
        ;;
    3)
        # Download ColBERT model only
        download_model "colbert-ir/colbertv2.0" "$COLBERT_DIR" "colbert"
        ;;
    4)
        # Download LLM only
        download_model "deepseek-ai/deepseek-coder-6.7b-instruct" "$LLM_DIR" "llm"
        ;;
    5)
        # Download Whisper only
        download_whisper "medium" "$WHISPER_DIR"
        ;;
    6)
        echo "Exiting without downloading any models."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Create .env file with model paths if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file with model paths..."
    cat > .env << EOF
# API settings
API_KEY=your-custom-api-key-here

# GPU settings
DEVICE=cuda:0
USE_FP16=true

# Model paths
EMBEDDING_MODEL=$(pwd)/$EMBEDDING_DIR
COLBERT_MODEL=$(pwd)/$COLBERT_DIR
DEEPSEEK_MODEL=$(pwd)/$LLM_DIR
WHISPER_MODEL_PATH=$(pwd)/$WHISPER_DIR
WHISPER_MODEL_SIZE=medium

# Additional settings
LLM_USE_4BIT=true
LLM_USE_8BIT=false
EOF
    echo ".env file created with model paths"
else
    echo ".env file already exists. You may need to update it manually with model paths."
    echo "Add the following lines to your .env file:"
    echo "EMBEDDING_MODEL=$(pwd)/$EMBEDDING_DIR"
    echo "COLBERT_MODEL=$(pwd)/$COLBERT_DIR"
    echo "DEEPSEEK_MODEL=$(pwd)/$LLM_DIR"
    echo "WHISPER_MODEL_PATH=$(pwd)/$WHISPER_DIR"
    echo "WHISPER_MODEL_SIZE=medium"
fi

echo "Model download process complete!"