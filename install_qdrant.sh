#!/bin/bash
# Script to install and run Qdrant locally without Docker

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=linux;;
    Darwin*)    OS_TYPE=macos;;
    MINGW*)     OS_TYPE=windows;;
    MSYS*)      OS_TYPE=windows;;
    *)          OS_TYPE="UNKNOWN:${OS}"
esac

# Set architecture
ARCH="$(uname -m)"
case "${ARCH}" in
    x86_64*)    ARCH_TYPE=x86_64;;
    amd64*)     ARCH_TYPE=x86_64;;
    arm64*)     ARCH_TYPE=aarch64;;
    aarch64*)   ARCH_TYPE=aarch64;;
    *)          ARCH_TYPE="UNKNOWN:${ARCH}"
esac

echo "Detected OS: $OS_TYPE, Architecture: $ARCH_TYPE"

# Set installation directory
INSTALL_DIR="qdrant"
mkdir -p $INSTALL_DIR

# Set Qdrant version
QDRANT_VERSION="v1.13.4"  # Change to latest version as needed
QDRANT_DATA_DIR="$INSTALL_DIR/data"
QDRANT_CONFIG_DIR="$INSTALL_DIR/config"

# Create data and config directories
mkdir -p $QDRANT_DATA_DIR
mkdir -p $QDRANT_CONFIG_DIR

# Download appropriate Qdrant binary
download_qdrant() {
    echo "Downloading Qdrant $QDRANT_VERSION for $OS_TYPE/$ARCH_TYPE..."

    if [ "$OS_TYPE" = "linux" ]; then
        if [ "$ARCH_TYPE" = "x86_64" ]; then
            DOWNLOAD_URL="https://github.com/qdrant/qdrant/releases/download/$QDRANT_VERSION/qdrant-linux-x86_64.tar.gz"
        elif [ "$ARCH_TYPE" = "aarch64" ]; then
            DOWNLOAD_URL="https://github.com/qdrant/qdrant/releases/download/$QDRANT_VERSION/qdrant-linux-aarch64.tar.gz"
        else
            echo "Unsupported architecture: $ARCH_TYPE"
            exit 1
        fi
    elif [ "$OS_TYPE" = "macos" ]; then
        if [ "$ARCH_TYPE" = "x86_64" ]; then
            DOWNLOAD_URL="https://github.com/qdrant/qdrant/releases/download/$QDRANT_VERSION/qdrant-macos-x86_64.tar.gz"
        elif [ "$ARCH_TYPE" = "aarch64" ]; then
            DOWNLOAD_URL="https://github.com/qdrant/qdrant/releases/download/$QDRANT_VERSION/qdrant-macos-aarch64.tar.gz"
        else
            echo "Unsupported architecture: $ARCH_TYPE"
            exit 1
        fi
    elif [ "$OS_TYPE" = "windows" ]; then
        echo "For Windows, please download Qdrant from: https://github.com/qdrant/qdrant/releases/download/$QDRANT_VERSION/qdrant-windows-x86_64.zip"
        echo "Extract it and run qdrant.exe manually."
        exit 1
    else
        echo "Unsupported OS: $OS_TYPE"
        exit 1
    fi

    # Download and extract
    TEMP_FILE="$INSTALL_DIR/qdrant.tar.gz"
    curl -L $DOWNLOAD_URL -o $TEMP_FILE

    if [ $? -ne 0 ]; then
        echo "Failed to download Qdrant. Please check your internet connection and try again."
        exit 1
    fi

    tar -xzf $TEMP_FILE -C $INSTALL_DIR
    rm $TEMP_FILE

    # Make qdrant executable
    chmod +x $INSTALL_DIR/qdrant

    echo "Qdrant downloaded successfully!"
}

# Create a basic config file
create_config() {
    echo "Creating Qdrant configuration..."

    cat > $QDRANT_CONFIG_DIR/config.yaml << EOL
storage:
  storage_path: ./data

service:
  http_port: 6333
  grpc_port: 6334
EOL

    echo "Configuration created successfully!"
}

# Function to run Qdrant
run_qdrant() {
    echo "Starting Qdrant..."
    cd $INSTALL_DIR
    ./qdrant --config config/config.yaml
}

# Update .env file to use local Qdrant
update_env() {
    echo "Updating .env file to use local Qdrant..."

    if [ -f .env ]; then
        # Update existing .env file
        sed -i.bak 's/QDRANT_HOST=.*/QDRANT_HOST=localhost/' .env
        sed -i.bak 's/QDRANT_PORT=.*/QDRANT_PORT=6333/' .env
        rm -f .env.bak
    elif [ -f .env.example ]; then
        # Create .env from .env.example
        cp .env.example .env
        sed -i.bak 's/QDRANT_HOST=.*/QDRANT_HOST=localhost/' .env
        sed -i.bak 's/QDRANT_PORT=.*/QDRANT_PORT=6333/' .env
        rm -f .env.bak
    else
        # Create new .env file
        echo "QDRANT_HOST=localhost" >> .env
        echo "QDRANT_PORT=6333" >> .env
    fi

    echo ".env file updated to use localhost:6333 for Qdrant."
}

# Main execution
case "$1" in
    install)
        download_qdrant
        create_config
        update_env
        echo "Installation complete. Run './install_qdrant.sh start' to start Qdrant."
        ;;
    start)
        if [ ! -f "$INSTALL_DIR/qdrant" ]; then
            echo "Qdrant not found. Please run './install_qdrant.sh install' first."
            exit 1
        fi
        run_qdrant
        ;;
    update-env)
        update_env
        ;;
    *)
        echo "Usage: $0 {install|start|update-env}"
        echo "  install    - Download and install Qdrant"
        echo "  start      - Start Qdrant server"
        echo "  update-env - Update .env file to use local Qdrant"
        ;;
esac

exit 0