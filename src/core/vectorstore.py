from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range, PointIdsList


class QdrantStore:
    """
    Enhanced Qdrant vector store that supports hybrid search with metadata filtering.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding_function: Embeddings,
    ):
        """
        Initialize the Qdrant vector store.

        Args:
            client: QdrantClient instance
            collection_name: Name of the collection to use
            embedding_function: Function to create embeddings
        """
        self.client = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function

        # Ensure collection exists
        self._ensure_collection()

        # Initialize Langchain Qdrant wrapper
        self.langchain_qdrant = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_function,
            distance=rest.Distance.COSINE,
        )

    def _ensure_collection(self) -> None:
        """
        Ensure the collection exists in Qdrant, creating it if necessary.
        """
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            # Get embedding dimension
            # Create a sample embedding to determine dimension
            sample_embedding = self.embedding_function.embed_query("sample text")
            embedding_dimension = len(sample_embedding)

            # Create the collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=embedding_dimension,
                    distance=rest.Distance.COSINE,
                ),
            )
            print(f"✅ QDrant collection {self.collection_name} created with {embedding_dimension} dimensions!")

            # Create payload index for common metadata fields
            self._create_payload_indexes()

    def _create_payload_indexes(self) -> None:
        """
        Create payload indexes for common metadata fields to speed up filtering.
        """
        common_fields = [
            "metadata.manufacturer",
            "metadata.model",
            "metadata.year",
            "metadata.category",
            "metadata.engine_type",
            "metadata.transmission",
            "metadata.source",
        ]

        for field in common_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs
        """
        return self.langchain_qdrant.add_documents(documents)

    def _build_filter(
        self, metadata_filter: Dict[str, Union[str, List[str], int, List[int]]]
    ) -> Filter:
        """
        Build a Qdrant filter from metadata filters.

        Args:
            metadata_filter: Dictionary of metadata filters

        Returns:
            Qdrant Filter object
        """
        must_conditions = []

        for key, value in metadata_filter.items():
            field_path = f"metadata.{key}"

            if isinstance(value, list):
                # For list values, create an OR condition
                should_conditions = []
                for v in value:
                    should_conditions.append(
                        FieldCondition(
                            key=field_path,
                            match=MatchValue(value=v),
                        )
                    )
                must_conditions.append(
                    Filter(
                        should=should_conditions,
                    )
                )
            elif isinstance(value, (int, float)) and key == "year":
                # For year ranges
                must_conditions.append(
                    FieldCondition(
                        key=field_path,
                        range=Range(
                            gte=value,
                            lte=value,
                        ),
                    )
                )
            else:
                # For single values
                must_conditions.append(
                    FieldCondition(
                        key=field_path,
                        match=MatchValue(value=value),
                    )
                )

        return Filter(must=must_conditions)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search with optional metadata filtering.

        Args:
            query: Query string
            k: Number of results to return
            metadata_filter: Optional metadata filters

        Returns:
            List of (document, score) tuples
        """
        if metadata_filter:
            filter_obj = self._build_filter(metadata_filter)
            return self.langchain_qdrant.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_obj,
            )
        else:
            return self.langchain_qdrant.similarity_search_with_score(
                query=query,
                k=k,
            )

    def delete_by_ids(self, ids: List[str]) -> None:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(
                points=ids,
            ),
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.

        Returns:
            Dictionary of collection statistics
        """
        return self.client.get_collection(self.collection_name).model_dump()
