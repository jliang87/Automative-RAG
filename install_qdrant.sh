#!/bin/bash
# Script to install and run Qdrant locally without Docker
# With support for Chinese mirrors

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
QDRANT_VERSION="v1.7.4"  # Change to latest version as needed
QDRANT_DATA_DIR="$INSTALL_DIR/data"
QDRANT_CONFIG_DIR="$INSTALL_DIR/config"

# Set mirror to use - options: github, gitee, tsinghua, aliyun
# Change this to your preferred mirror, or set the QDRANT_MIRROR environment variable
# Example: export QDRANT_MIRROR=gitee
DEFAULT_MIRROR="gitee"  # Default to Gitee for China

# Create data and config directories
mkdir -p $QDRANT_DATA_DIR
mkdir -p $QDRANT_CONFIG_DIR

# Download appropriate Qdrant binary
download_qdrant() {
    echo "Downloading Qdrant $QDRANT_VERSION for $OS_TYPE/$ARCH_TYPE..."

    # Choose mirror based on environment variable or default
    MIRROR=${QDRANT_MIRROR:-$DEFAULT_MIRROR}

    # Set base URL based on mirror
    if [ "$MIRROR" = "gitee" ]; then
        # Gitee mirror (China)
        BASE_URL="https://gitee.com/mirrors/qdrant/releases/download/$QDRANT_VERSION"
        echo "Using Gitee mirror for download"
    elif [ "$MIRROR" = "tsinghua" ]; then
        # Tsinghua mirror (China)
        BASE_URL="https://mirrors.tuna.tsinghua.edu.cn/github-release/qdrant/qdrant/$QDRANT_VERSION"
        echo "Using Tsinghua mirror for download"
    elif [ "$MIRROR" = "aliyun" ]; then
        # Aliyun mirror (China)
        BASE_URL="https://mirrors.aliyun.com/github-release/qdrant/qdrant/releases/download/$QDRANT_VERSION"
        echo "Using Aliyun mirror for download"
    else
        # Default GitHub
        BASE_URL="https://github.com/qdrant/qdrant/releases/download/$QDRANT_VERSION"
        echo "Using GitHub for download"
    fi

    if [ "$OS_TYPE" = "linux" ]; then
        if [ "$ARCH_TYPE" = "x86_64" ]; then
            FILENAME="qdrant-linux-x86_64.tar.gz"
        elif [ "$ARCH_TYPE" = "aarch64" ]; then
            FILENAME="qdrant-linux-aarch64.tar.gz"
        else
            echo "Unsupported architecture: $ARCH_TYPE"
            exit 1
        fi
    elif [ "$OS_TYPE" = "macos" ]; then
        if [ "$ARCH_TYPE" = "x86_64" ]; then
            FILENAME="qdrant-macos-x86_64.tar.gz"
        elif [ "$ARCH_TYPE" = "aarch64" ]; then
            FILENAME="qdrant-macos-aarch64.tar.gz"
        else
            echo "Unsupported architecture: $ARCH_TYPE"
            exit 1
        fi
    elif [ "$OS_TYPE" = "windows" ]; then
        echo "For Windows, please download Qdrant manually from a mirror site."
        echo "Extract it and run qdrant.exe manually."
        exit 1
    else
        echo "Unsupported OS: $OS_TYPE"
        exit 1
    fi

    # Construct full download URL
    DOWNLOAD_URL="$BASE_URL/$FILENAME"
    echo "Download URL: $DOWNLOAD_URL"

    # Download and extract
    TEMP_FILE="$INSTALL_DIR/qdrant.tar.gz"

    # Try different download methods
    echo "Downloading from: $DOWNLOAD_URL"
    if command -v curl &> /dev/null; then
        curl -L $DOWNLOAD_URL -o $TEMP_FILE
        DOWNLOAD_STATUS=$?
    elif command -v wget &> /dev/null; then
        wget $DOWNLOAD_URL -O $TEMP_FILE
        DOWNLOAD_STATUS=$?
    else
        echo "Neither curl nor wget found. Please install one of them and try again."
        exit 1
    fi

    if [ $DOWNLOAD_STATUS -ne 0 ]; then
        echo "Failed to download Qdrant."
        echo "If you're in China, try setting a different mirror:"
        echo "  export QDRANT_MIRROR=gitee    # Gitee mirror"
        echo "  export QDRANT_MIRROR=tsinghua # Tsinghua mirror"
        echo "  export QDRANT_MIRROR=aliyun   # Aliyun mirror"
        echo "Or download manually from a mirror and place in $INSTALL_DIR"
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
if [ "$1" = "help" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ] || [ $# -eq 0 ]; then
    echo "Usage: $0 {install|start|update-env|help} [mirror]"
    echo "  install    - Download and install Qdrant"
    echo "  start      - Start Qdrant server"
    echo "  update-env - Update .env file to use local Qdrant"
    echo "  help       - Show this help message"
    echo ""
    echo "Optional mirror parameter (for install):"
    echo "  gitee     - Use Gitee mirror (China)"
    echo "  tsinghua  - Use Tsinghua mirror (China)"
    echo "  aliyun    - Use Aliyun mirror (China)"
    echo "  github    - Use GitHub (default, may be blocked in China)"
    echo ""
    echo "You can also set the mirror using environment variable:"
    echo "  export QDRANT_MIRROR=gitee"
    echo ""
    exit 0
fi

# Check for mirror parameter
if [ ! -z "$2" ]; then
    export QDRANT_MIRROR="$2"
    echo "Setting mirror to: $QDRANT_MIRROR"
fi

# Main command execution
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
        echo "Unknown command: $1"
        echo "Use './install_qdrant.sh help' for usage information."
        exit 1
        ;;
esac

exit 0