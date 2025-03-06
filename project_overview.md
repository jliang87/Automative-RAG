# Automotive Specs RAG System

A production-grade AI workflow for automotive specifications using hybrid search, late interaction RAG with ColBERT reranking, and DeepSeek LLM.

## Architecture Overview

![Architecture Diagram]

### Components

1. **Data Ingestion Pipeline**
   - YouTube video transcription
   - PDF parsing
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
- **Multi-modal Input**: Process both YouTube videos and PDFs
- **Source Attribution**: Track provenance of information through the pipeline
- **Automotive Domain Specialization**: Optimized for automotive specifications
- **Production-Ready**: Containerized with proper dependency management and API documentation
