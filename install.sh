#!/bin/bash
# install.sh

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "Failed to install Poetry. Please install it manually."
    echo "https://python-poetry.org/docs/#installation"
    exit 1
fi

echo "Installing dependencies with Poetry..."
poetry install

# Create necessary directories
echo "Creating data directories..."
mkdir -p data/uploads data/youtube data/bilibili data/pdfs models/cache models/embeddings models/llm models/whisper

echo "Installation complete!"
echo "---------------------------------------------"
echo "You can now run the application using:"
echo "  ./run_api.sh    # Start the API server"
echo "  ./run_ui.sh     # Start the Streamlit UI"
echo "---------------------------------------------"