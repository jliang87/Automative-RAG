# Local DeepSeek LLM Integration

This document explains how to use the locally-hosted DeepSeek model in the Automotive Specs RAG system, leveraging your GPU for all components of the pipeline.

## Overview

The system now uses a local DeepSeek model for inference instead of relying on the DeepSeek API. This provides several advantages:

1. **Complete GPU Acceleration**: The entire pipeline runs on your GPU, eliminating API dependencies
2. **Cost Savings**: No API usage fees or tokens to manage
3. **Privacy**: All data processing happens locally on your infrastructure
4. **Customization**: Full control over model configuration and inference parameters
5. **Offline Capability**: The system can operate without internet connectivity

## Supported Models

The system supports various DeepSeek models including:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (recommended balance of performance/VRAM)
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (for lower VRAM requirements)
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` (highest quality, requires >24GB VRAM)

## Hardware Requirements

To run the local LLM effectively, you'll need:

- NVIDIA GPU with at least 12GB VRAM (16GB+ recommended for 6.7B model)
- 32GB+ system RAM
- SSD storage for model files (15-30GB depending on model size)

## Memory Optimization

The system uses several techniques to reduce VRAM usage:

1. **4-bit Quantization**: Reduces model size by 75% with minimal quality impact
2. **8-bit Alternative**: Optional 8-bit quantization if 4-bit causes issues
3. **Flash Attention**: Efficient attention implementation that requires less memory
4. **Gradient Checkpointing**: Reduces memory footprint during inference
5. **Mixed Precision**: Uses FP16 for faster computation

## Configuration Options

You can configure the local LLM through the `.env` file or environment variables:

```
# Model Selection
DEEPSEEK_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# Quantization Options
LLM_USE_4BIT=true  # Enable 4-bit quantization
LLM_USE_8BIT=false  # Enable 8-bit quantization (alternative to 4-bit)

# Inference Settings
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=512

# Cache Directories
TRANSFORMERS_CACHE=/app/models/cache
HF_HOME=/app/models/hub
```

## Performance Comparison

| Model Size | Quantization | VRAM Usage | Generation Speed | Quality |
|------------|--------------|------------|------------------|---------|
| 1.3B       | 4-bit        | ~2GB       | ~15 tokens/sec   | ★★☆☆☆   |
| 6.7B       | 4-bit        | ~8GB       | ~10 tokens/sec   | ★★★★☆   |
| 6.7B       | 8-bit        | ~12GB      | ~12 tokens/sec   | ★★★★☆   |
| 33B        | 4-bit        | ~24GB      | ~5 tokens/sec    | ★★★★★   |

## First-Time Setup

When first running the system, it will automatically download the model from Hugging Face. This may take some time depending on your internet connection. The model is cached for subsequent runs.

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. Use a smaller model (1.3B instead of 6.7B)
2. Enable 4-bit quantization (if not already enabled)
3. Reduce `LLM_MAX_TOKENS` to limit generation length
4. Try setting `device_map="auto"` in the model configuration to distribute across multiple GPUs if available

### Generation Quality Issues

If you notice poor generation quality:

1. Try 8-bit quantization instead of 4-bit
2. Increase the `LLM_TEMPERATURE` slightly (0.1-0.3)
3. If you have sufficient VRAM, try disabling quantization entirely

## Monitoring

You can monitor the LLM's performance and resource usage via the new endpoint:

```
GET /query/llm-info
```

This endpoint returns details about the model configuration, device usage, and VRAM consumption.

## Example Usage

To query the local LLM directly:

```python
from src.core.llm import LocalLLM

# Initialize with GPU support
llm = LocalLLM(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    device="cuda:0",
    temperature=0.1,
    max_tokens=512,
    use_4bit=True,
    use_8bit=False
)

# Generate a response
query = "What is the horsepower of the 2023 Toyota Camry?"
documents = [...]  # Your retrieved documents
answer = llm.answer_query(query, documents)
```
