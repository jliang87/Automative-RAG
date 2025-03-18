from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient("localhost", port=6333)

client.create_collection(
    collection_name="automotive_specs",
    vectors_config=VectorParams(
        size=1024,  # Dimension of bge-large-zh-v1.5 embeddings
        distance=Distance.DOT  # Best for normalized embeddings
    )
)