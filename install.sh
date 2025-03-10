#!/bin/bash
# Script to install dependencies for the Automotive Specs RAG system

echo "Installing dependencies for Automotive Specs RAG system..."

# Check if .env file exists, if not create from .env.example
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "Created .env file from .env.example"
    else
        echo "Error: .env.example not found"
        exit 1
    fi
fi

# Check for Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+ and try again."
    exit 1
fi

# Check Python version
python_version=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
python_min_version="3.8"

if [ "$(printf '%s\n' "$python_min_version" "$python_version" | sort -V | head -n1)" != "$python_min_version" ]; then
    echo "Error: Python $python_min_version or higher is required. Found Python $python_version"
    exit 1
fi

echo "Using Python $python_version"

# Check if Poetry is installed
if command -v poetry &> /dev/null; then
    echo "Poetry found, installing dependencies..."
    poetry install
else
    echo "Poetry not found, using pip to install dependencies..."

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python -m venv venv
    fi

    # Activate virtual environment
    echo "Activating virtual environment..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        source venv/Scripts/activate
    else
        # Linux/Mac
        source venv/bin/activate
    fi

    # Install dependencies
    echo "Installing dependencies from requirements.txt..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        echo "Warning: requirements.txt not found. Generating from pyproject.toml..."
        if [ -f "pyproject.toml" ]; then
            # If poetry is not installed but pyproject.toml exists, try to extract dependencies
            echo "Installing basic dependencies first..."
            pip install torch langchain fastapi uvicorn streamlit transformers sentence-transformers qdrant-client

            echo "Please run download_models.sh separately to download required models."
        else
            echo "Error: Neither requirements.txt nor pyproject.toml found."
            exit 1
        fi
    fi
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/uploads data/youtube data/bilibili models/embeddings models/colbert models/llm models/whisper models/cache models/hub

echo "Installation complete!"
echo "Next steps:"
echo "1. Review the .env file and adjust settings if needed"
echo "2. Run download_models.sh to download necessary models"
echo "3. Start the API server with run_api.sh"
echo "4. Start the UI server with run_ui.sh"
echo ""
echo "GPU status: $(python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')")"