#!/bin/bash
# Script to run the Streamlit UI

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

# Load environment variables from .env file
set -a
source .env
set +a

# Set default values if not specified in .env
UI_PORT=${UI_PORT:-"8501"}
API_URL=${API_URL:-"http://localhost:8000"}
API_KEY=${API_KEY:-"default-api-key"}

echo "Starting Streamlit UI on port ${UI_PORT}"
echo "Connecting to API at ${API_URL}"

# Check for Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+ and try again."
    exit 1
fi

# Check for required packages
if ! python -c "import streamlit" &> /dev/null; then
    echo "Error: streamlit not found. Please install it with: pip install streamlit"
    exit 1
fi

# Set environment variables for Streamlit
export API_URL=$API_URL
export API_KEY=$API_KEY

# Run Streamlit app
streamlit run src/ui/app.py --server.port=$UI_PORT --server.address=0.0.0.0

# Exit code
exit $?