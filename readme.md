## Quick Start

### Standard Installation (Global Access)

```bash
# Make scripts executable
chmod +x install.sh run_api.sh run_ui.sh download_models.sh

# Install dependencies
./install.sh

# Download required models
./download_models.sh

# Run the application in two separate terminals
./run_api.sh  # in one terminal
./run_ui.sh   # in another terminal
```

### Installation in China (Where Hugging Face is blocked)

If you're in a region where Hugging Face is blocked, use the China-specific download script:

```bash
# Make scripts executable
chmod +x install.sh run_api.sh run_ui.sh download_models_cn.sh

# Install dependencies using mirrors
pip install -i https://mirror.baidu.com/pypi/simple/ -r requirements.txt

# Download required models using China mirrors
./download_models_cn.sh

# Run the application in two separate terminals
./run_api.sh  # in one terminal
./run_ui.sh   # in another terminal
```

### Docker Installation

For a containerized setup:

```bash
# Copy environment file and modify as needed
cp .env.example .env

# Edit .env with your preferred settings
nano .env

# Build and start the containers
docker-compose up -d

# Access the UI at http://localhost:8501
# API runs on http://localhost:8000
```
# Running Qdrant Locally

This guide explains how to run Qdrant vector database locally without Docker.

## Using the Installation Script

We provide a script that automates the installation and running of Qdrant:

```bash
# Make the script executable
chmod +x install_qdrant.sh

# Install Qdrant
./install_qdrant.sh install

# Start Qdrant
./install_qdrant.sh start
```

The script will:
1. Download the appropriate Qdrant binary for your OS and architecture
2. Create a basic configuration file
3. Update your `.env` file to connect to the local Qdrant instance
4. Provide a command to start the Qdrant server

## Manual Installation

If you prefer to install Qdrant manually:

1. Download the appropriate binary for your system from the [Qdrant releases page](https://github.com/qdrant/qdrant/releases)

2. Extract the archive:
   ```bash
   tar -xzf qdrant-*.tar.gz
   ```
   
3. Make the binary executable:
   ```bash
   chmod +x qdrant
   ```

4. Create a configuration file:
   ```bash
   mkdir -p config
   cat > config/config.yaml << EOL
   storage:
     storage_path: ./data
     
   service:
     http_port: 6333
     grpc_port: 6334
   EOL
   ```

5. Run Qdrant:
   ```bash
   ./qdrant --config config/config.yaml
   ```

6. Update your `.env` file to use localhost:
   ```
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   ```

## Verifying Installation

To verify that Qdrant is running correctly:

```bash
curl http://localhost:6333/cluster
```

You should see something like:
```json
{
  "status": "ok",
  "result": {
    "status": "green",
    ...
  }
}
```
# Running Qdrant Locally (China-friendly)

This guide explains how to run Qdrant vector database locally without Docker, with special support for users in China who may have difficulty accessing GitHub.

## Using the Installation Script with Chinese Mirrors

We provide a script that automates the installation and running of Qdrant, with support for Chinese mirrors:

```bash
# Make the script executable
chmod +x install_qdrant.sh

# Install Qdrant using Gitee mirror (best for China)
./install_qdrant.sh install gitee

# Alternatively, you can use other mirrors:
# ./install_qdrant.sh install tsinghua  # Tsinghua University mirror
# ./install_qdrant.sh install aliyun    # Aliyun mirror

# Start Qdrant
./install_qdrant.sh start
```

You can also set the mirror using an environment variable:
```bash
export QDRANT_MIRROR=gitee
./install_qdrant.sh install
```

The script will:
1. Download the Qdrant binary from your chosen mirror
2. Create a basic configuration file
3. Update your `.env` file to connect to the local Qdrant instance
4. Provide a command to start the Qdrant server

## Manual Installation for China Users

If you prefer to install Qdrant manually:

1. Download the appropriate binary from a mirror site:
   - Gitee: https://gitee.com/mirrors/qdrant/releases/
   - Or download through a proxy/VPN

2. Extract the archive:
   ```bash
   tar -xzf qdrant-*.tar.gz
   ```
   
3. Make the binary executable:
   ```bash
   chmod +x qdrant
   ```

4. Create a configuration file:
   ```bash
   mkdir -p config
   cat > config/config.yaml << EOL
   storage:
     storage_path: ./data
     
   service:
     http_port: 6333
     grpc_port: 6334
   EOL
   ```

5. Run Qdrant:
   ```bash
   ./qdrant --config config/config.yaml
   ```

6. Update your `.env` file to use localhost:
   ```
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   ```

## Verifying Installation

To verify that Qdrant is running correctly:

```bash
curl http://localhost:6333/cluster
```

You should see something like:
```json
{
  "status": "ok",
  "result": {
    "status": "green",
    ...
  }
}
```

You can now run your application and it will connect to the local Qdrant instance.

You can now run your application and it will connect to the local Qdrant instance.

## Model Configuration

The system uses the following AI models:

- **Embedding Model**: BAAI/bge-small-en-v1.5
- **Retrieval Model**: colbert-ir/colbertv2.0
- **Language Model**: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- **Speech-to-Text**: OpenAI Whisper (medium)

You can configure model paths and other settings in the `.env` file.

# GPU Acceleration in Automotive Specs RAG

This document outlines the GPU acceleration capabilities in the Automotive Specs RAG system and how to take full advantage of them.

## Overview of GPU-Accelerated Components

The system leverages GPU acceleration for several key components:

1. **YouTube/Bilibili Transcription**: Uses Whisper with CUDA acceleration
2. **PDF Processing**: GPU-accelerated OCR and table extraction
3. **ColBERT Reranking**: Token-level matching with batched processing
4. **Vector Embeddings**: GPU-accelerated embedding generation

## Hardware Requirements

- NVIDIA GPU with CUDA support (at least 8GB VRAM recommended)
- CUDA 11.8+ and cuDNN installed on the host machine
- Docker with NVIDIA Container Toolkit installed

## Performance Benefits

| Component | CPU Performance | GPU Performance (RTX 3080) | Speedup |
|-----------|----------------|----------------------------|---------|
| Whisper Transcription (5min video) | ~12 minutes | ~40 seconds | 18x |
| ColBERT Reranking (100 docs) | ~30 seconds | ~2 seconds | 15x |
| PDF OCR (50-page document) | ~5 minutes | ~30 seconds | 10x |
| Vector Embedding Generation | ~10 docs/sec | ~120 docs/sec | 12x |

## Configuration Options

The GPU acceleration can be configured in the `.env` file:

```
# GPU Settings
DEVICE=cuda  # Use 'cpu' to disable GPU
USE_FP16=true  # Enable mixed precision (faster but slightly less accurate)
COLBERT_BATCH_SIZE=16  # Increase for faster processing, decrease if OOM errors occur

# Whisper Settings
WHISPER_MODEL_SIZE=medium  # Options: tiny, base, small, medium, large
USE_YOUTUBE_CAPTIONS=true  # Try YouTube captions first (faster)
USE_WHISPER_AS_FALLBACK=true  # Use Whisper if captions aren't available

# PDF OCR Settings
USE_PDF_OCR=true  # Enable GPU-accelerated OCR for scanned PDFs
OCR_LANGUAGES=eng  # Language codes for OCR
```

## Memory Optimization

The system includes several memory optimization techniques:

1. **Batch Processing**: Documents are processed in batches to control memory usage
2. **Mixed Precision (FP16)**: Reduces memory usage by half with minimal quality impact
3. **Model Offloading**: Models are loaded only when needed and unloaded after use
4. **Lazy Loading**: Whisper models are loaded on-demand to save memory

## Monitoring GPU Usage

You can monitor GPU usage during operation with:

```bash
# Inside the Docker container
nvidia-smi -l 1

# Or from the host
docker exec -it auto-specs-rag-api-1 nvidia-smi -l 1
```

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory errors:

1. Reduce batch sizes (COLBERT_BATCH_SIZE)
2. Use a smaller Whisper model (base instead of medium)
3. Enable mixed precision (USE_FP16=true)
4. Reduce the maximum document length for ColBERT

### Poor Performance

If GPU acceleration isn't providing expected speedups:

1. Verify CUDA is available: `docker exec -it auto-specs-rag-api-1 python -c "import torch; print(torch.cuda.is_available())"`
2. Check if models are using GPU: `docker logs auto-specs-rag-api-1 | grep "Using GPU"`
3. Ensure the GPU isn't being throttled: `nvidia-smi -q -d PERFORMANCE`

## Example: Processing YouTube Videos with GPU

```python
from src.core.youtube_transcriber import YouTubeTranscriber

# Initialize with GPU support
transcriber = YouTubeTranscriber(
    whisper_model_size="medium",  # Options: tiny, base, small, medium, large
    device="cuda",  # Will default to CUDA if available
    use_youtube_captions=True,  # Try YouTube captions first (faster)
    use_whisper_as_fallback=True  # Use Whisper if captions unavailable
)

# Process a single video with GPU acceleration
documents = transcriber.process_video(
    url="https://www.youtube.com/watch?v=example",
    force_whisper=True  # Force using Whisper even if captions exist
)
```

## Example: GPU-Accelerated PDF Processing

```python
from src.core.pdf_loader import PDFLoader

# Initialize with GPU support
pdf_loader = PDFLoader(
    device="cuda",
    use_ocr=True,  # Enable GPU-accelerated OCR
    ocr_languages="eng"  # Set OCR languages
)

# Process a PDF with GPU-accelerated OCR and extraction
documents = pdf_loader.process_pdf(
    file_path="path/to/document.pdf",
    extract_tables=True  # Extract tables using GPU acceleration
)
```
