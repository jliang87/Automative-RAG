auto-specs-rag/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.streamlit
├── pyproject.toml
├── README.md
├── scripts/
│   ├── load_example_data.py
│   └── test_end_to_end.py
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── dependencies.py
│   │   └── routers/
│   │       ├── __init__.py
│   │       ├── auth.py
│   │       ├── ingest.py
│   │       └── query.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── colbert_reranker.py
│   │   ├── document_processor.py
│   │   ├── llm.py
│   │   ├── pdf_loader.py
│   │   ├── retriever.py
│   │   ├── vectorstore.py
│   │   └── youtube_transcriber.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── components.py
│   │   └── pages/
│   │       ├── __init__.py
│   │       ├── home.py
│   │       ├── ingest.py
│   │       └── query.py
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py
│       └── logging.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_api.py
    ├── test_colbert.py
    ├── test_ingest.py
    └── test_retrieval.py
