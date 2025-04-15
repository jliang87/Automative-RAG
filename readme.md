# Automotive Specs RAG System Setup

This guide explains how to set up and run the Automotive Specs RAG system.

## Prerequisites

- Python 3.8+ for local setup
- Docker and Docker Compose for containerized setup
- NVIDIA GPU with CUDA support for GPU acceleration
- If in China: Access to Chinese mirrors or a reliable VPN

## Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd <repo-directory>
```

## Step 2: Setup Configuration

```bash
# Copy example environment file
cp .env.example .env
```

Edit the `.env` file to adjust settings if needed.

## Step 3: Download Models (REQUIRED)

This system requires pre-downloaded models to function properly. Choose the appropriate download script:

### Standard Download (Global)

```bash
# Make the script executable
chmod +x download_models.sh

# Download all models
./download_models.sh
```

This process may take some time, especially for the large language model.

## Step 4: Start the System

You can run the system either locally or using Docker.

### Option A: Run with Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# Check logs if needed
docker-compose logs -f api
```

The services will be available at:
- API: http://localhost:8000
- UI: http://localhost:8501
- Qdrant: http://localhost:6333

### Option B: Run Locally

#### Step 4B.1: Install Qdrant

```bash
# Make the script executable
chmod +x install_qdrant.sh

# Install Qdrant
./install_qdrant.sh install

# Start Qdrant in a separate terminal
./install_qdrant.sh start
```

#### Step 4B.2: Run the API and UI

In separate terminals:

```bash
# Terminal 1: Start the API
chmod +x run_api.sh
./run_api.sh

# Terminal 2: Start the UI
chmod +x run_ui.sh
./run_ui.sh
```

## Verifying Installation

1. Check that all services are running:
   - API should be accessible at http://localhost:8000/docs
   - UI should be accessible at http://localhost:8501
   - Qdrant should respond at http://localhost:6333/dashboard

2. Try ingesting a sample document through the UI or API.

3. Run a test query to ensure the whole pipeline works.

## System Architecture

### Components

1. **Data Ingestion Pipeline**
   - Video transcription (YouTube, Bilibili) using Whisper AI
   - PDF parsing with OCR capabilities
   - Metadata extraction
   - Document chunking

2. **Vector Database**
   - Qdrant for vector storage
   - Hybrid search (vector + metadata filtering)
   - Document storage

3. **Retrieval System**
   - Initial retrieval via Qdrant
   - ColBERT reranking for precise semantic matching
   - Late interaction patterns for token-level analysis

4. **Generation Layer**
   - DeepSeek LLM integration
   - Context assembly with metadata
   - Response generation

5. **API Layer**
   - FastAPI backend
   - Swagger documentation
   - Authentication

6. **User Interface**
   - Streamlit dashboard
   - Query input
   - Results visualization
   - Source attribution

7. **Deployment Infrastructure**
   - Docker containerization
   - Docker Compose orchestration
   - GPU support
   - Poetry dependency management

## Key Features

- **Hybrid Search**: Combine vector similarity search with metadata filtering
- **Late Interaction Retrieval**: ColBERT reranking for high-precision document matching
- **Multi-modal Input**: Process both videos and PDFs
- **Unified Video Processing**: Process videos from multiple platforms (YouTube, Bilibili) with Whisper AI transcription
- **Source Attribution**: Track provenance of information through the pipeline
- **Automotive Domain Specialization**: Optimized for automotive specifications
- **GPU Acceleration**: All components leverage GPU for maximum performance
- **Production-Ready**: Containerized with proper dependency management and API documentation

## Model Configuration

The system uses the following AI models:

- **Embedding Model**: BAAI/bge-m3
- **Retrieval Model**: colbert-ir/colbertv2.0
- **Language Model**: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- **Speech-to-Text**: OpenAI Whisper (medium)

You can configure model paths and other settings in the `.env` file.

## GPU Acceleration

This system leverages GPU acceleration for several key components:

1. **Video Transcription**: Uses Whisper with CUDA acceleration
2. **PDF Processing**: GPU-accelerated OCR and table extraction
3. **ColBERT Reranking**: Token-level matching with batched processing
4. **Vector Embeddings**: GPU-accelerated embedding generation

For best performance, a NVIDIA GPU with at least 8GB VRAM is recommended.

## Troubleshooting

### Model Loading Issues

If you encounter model loading errors:

1. Ensure you've run the model download script
2. Check the model directories under `./models/`
3. Verify that Docker volumes are correctly mounted (if using Docker)

### API Connection Issues

If the UI can't connect to the API:

1. Check that the API is running
2. Verify the `API_URL` environment variable is correctly set
3. Check network connectivity between services

### GPU Issues

If GPU acceleration isn't working:

1. Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. Check that the GPU is visible to Docker (if using Docker)
3. Ensure the correct device is specified in `.env`