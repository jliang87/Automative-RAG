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
