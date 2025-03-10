#!/bin/bash
# Script to run the FastAPI backend server

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
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8000"}
LOG_LEVEL=${LOG_LEVEL:-"info"}

echo "Starting API server on ${HOST}:${PORT} with log level ${LOG_LEVEL}"
echo "GPU status: $(python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')")"

# Check for Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+ and try again."
    exit 1
fi

# Check for required packages
if ! python -c "import uvicorn" &> /dev/null; then
    echo "Error: uvicorn not found. Please install it with: pip install uvicorn[standard]"
    exit 1
fi

if ! python -c "import fastapi" &> /dev/null; then
    echo "Error: fastapi not found. Please install it with: pip install fastapi"
    exit 1
fi

# Run API server with uvicorn
uvicorn src.api.main:app --host $HOST --port $PORT --log-level $LOG_LEVEL --reload

# Exit code
exit $?