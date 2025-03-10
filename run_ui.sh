#!/bin/bash
# run_ui.sh

# Set Python path to include current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Set default API URL and port
export API_URL=${API_URL:-"http://localhost:8000"}

# Load API key from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    API_KEY=$(grep "API_KEY" .env | cut -d '=' -f2)
    if [ -n "$API_KEY" ]; then
        export API_KEY
    else
        export API_KEY="default-api-key"
    fi
else
    export API_KEY="default-api-key"
fi

echo "Connecting to API at: $API_URL"
echo "Starting Streamlit UI..."

# Start the Streamlit UI with Poetry
poetry run streamlit run src/ui/app.py