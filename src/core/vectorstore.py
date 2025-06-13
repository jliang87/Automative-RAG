from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import time
import numpy as np
from fastapi import Depends, HTTPException, Header, status
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, Range, PointIdsList

# Configure logging
logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Enhanced Qdrant vector store that supports hybrid search with metadata filtering,
    verification, and repair functionality.

    SIMPLIFIED: No more metadata-only mode - always requires embedding function.
    """

    def __init__(
            self,
            client: QdrantClient,
            collection_name: str,
            embedding_function: Embeddings,  # ✅ Always required now
    ):
        """
        Initialize the Qdrant vector store.

        Args:
            client: QdrantClient instance
            collection_name: Name of the collection to use
            embedding_function: Function to create embeddings (REQUIRED)
        """
        if embedding_function is None:
            raise ValueError("Embedding function is required. No more metadata-only mode.")

        self.client = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function

        logger.info(f"Initializing QdrantStore with collection: {collection_name}")

        # Ensure collection exists
        self._ensure_collection()

        # Initialize Langchain Qdrant wrapper
        self.langchain_qdrant = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding_function,
            distance=rest.Distance.COSINE,
        )
        logger.info(f"QdrantStore initialized successfully with embedding function")

    def _ensure_collection(self) -> None:
        """
        Ensure the collection exists in Qdrant, creating it if necessary.
        """
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            # Get embedding dimension
            sample_embedding = self.embedding_function.embed_query("sample text")
            embedding_dimension = len(sample_embedding)

            logger.info(f"Creating collection '{self.collection_name}' with {embedding_dimension} dimensions")

            # Create the collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(
                    size=embedding_dimension,
                    distance=rest.Distance.COSINE,
                ),
            )
            logger.info(f"Collection '{self.collection_name}' created successfully")

            # Create payload index for common metadata fields
            self._create_payload_indexes()
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

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
            "metadata.source_id",
            "metadata.ingestion_time",
        ]

        logger.info(f"Creating payload indexes for {len(common_fields)} fields")

        for field in common_fields:
            try:
                # Determine appropriate schema type based on field name
                if field.endswith("year") or field.endswith("_time"):
                    schema_type = rest.PayloadSchemaType.INTEGER
                else:
                    schema_type = rest.PayloadSchemaType.KEYWORD

                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema_type,
                )
                logger.info(f"Created index for {field} with schema type {schema_type}")
            except Exception as e:
                logger.warning(f"Failed to create index for {field}: {str(e)}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store with enhanced logging and validation.

        Args:
            documents: List of documents to add

        Returns:
            List of document IDs
        """
        if not documents:
            logger.warning("No documents provided to add_documents")
            return []

        logger.info(f"Adding {len(documents)} documents to collection '{self.collection_name}'")

        # Add ingestion timestamp if not present
        current_time = time.time()
        for doc in documents:
            if "ingestion_time" not in doc.metadata:
                doc.metadata["ingestion_time"] = current_time

        try:
            # Ensure document IDs exist for each document
            doc_ids = []
            for doc in documents:
                if "id" not in doc.metadata or not doc.metadata["id"]:
                    doc.metadata["id"] = f"doc-{str(time.time())}-{len(doc_ids)}"
                doc_ids.append(doc.metadata["id"])

            # Add documents to vector store using the embedding function
            result_ids = self.langchain_qdrant.add_documents(documents)

            # Verify points were added successfully
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection now has {collection_info.vectors_count} vectors")

            return result_ids
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    def similarity_search_with_score(
            self,
            query: str,
            k: int = 5,
            metadata_filter: Optional[Dict[str, Union[str, List[str], int, List[int]]]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search with optional metadata filtering.

        SIMPLIFIED: Always uses vector similarity - no metadata-only fallback.

        Args:
            query: Query string
            k: Number of results to return
            metadata_filter: Optional metadata filters

        Returns:
            List of (document, score) tuples
        """
        logger.info(f"Performing similarity search for query: '{query}' with k={k}")

        if metadata_filter:
            logger.info(f"Using metadata filter: {metadata_filter}")
            filter_obj = self._build_filter(metadata_filter)

            try:
                results = self.langchain_qdrant.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_obj,
                )
                logger.info(f"Search returned {len(results)} results with filter")
                return results
            except Exception as e:
                logger.error(f"Error in filtered search: {str(e)}, falling back to unfiltered search")
                # Fallback to unfiltered search if filter causes issues
                results = self.langchain_qdrant.similarity_search_with_score(
                    query=query,
                    k=k,
                )
                logger.info(f"Fallback search returned {len(results)} results")
                return results
        else:
            results = self.langchain_qdrant.similarity_search_with_score(
                query=query,
                k=k,
            )
            logger.info(f"Search returned {len(results)} results without filter")
            return results

    def _build_filter(
            self, metadata_filter: Dict[str, Union[str, List[str], int, List[int]]]
    ) -> Filter:
        """
        Build a Qdrant filter from metadata filters with enhanced validation.

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
                if not value:  # Skip empty lists
                    continue

                should_conditions = []
                for v in value:
                    if v is None:  # Skip None values
                        continue

                    should_conditions.append(
                        FieldCondition(
                            key=field_path,
                            match=MatchValue(value=v),
                        )
                    )

                if should_conditions:  # Only add if there are conditions
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
            elif value is not None:  # Skip None values
                # For single values
                must_conditions.append(
                    FieldCondition(
                        key=field_path,
                        match=MatchValue(value=value),
                    )
                )

        return Filter(must=must_conditions)

    def search_by_metadata(
            self, metadata_filter: Dict[str, Any], limit: int = 100
    ) -> List[Document]:
        """
        Search for documents by metadata only (no vector similarity).

        Args:
            metadata_filter: Metadata criteria to search for
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        logger.info(f"Searching by metadata: {metadata_filter}, limit={limit}")

        filter_obj = self._build_filter(metadata_filter)

        try:
            # Get matching points
            points = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_obj,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )[0]

            # Convert to documents
            documents = []
            for point in points:
                metadata = point.payload.get("metadata", {})
                content = point.payload.get("page_content", "")

                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

            logger.info(f"Metadata search returned {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error in metadata search: {str(e)}")
            return []

    def delete_by_ids(self, ids: List[str]) -> None:
        """
        Delete documents by IDs with enhanced validation and error handling.

        Args:
            ids: List of document IDs to delete
        """
        if not ids:
            logger.warning("No document IDs provided for deletion")
            return

        logger.info(f"Deleting {len(ids)} documents from collection '{self.collection_name}'")

        try:
            # Check collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                logger.error(f"Collection '{self.collection_name}' does not exist")
                return

            # Delete points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(
                    points=ids,
                ),
            )

            logger.info(f"Successfully deleted {len(ids)} documents")
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics with enhanced error handling.

        Returns:
            Dictionary of collection statistics
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            stats = collection_info.dict()

            # Add some additional useful info
            stats["name"] = self.collection_name

            # Check index status - use correct API method
            # The proper method is get_collection_info instead of get_collection_indices
            try:
                # In newer versions of Qdrant client, the index information is
                # already included in the collection_info
                if hasattr(collection_info, "payload_schema"):
                    stats["payload_indices"] = collection_info.payload_schema
                else:
                    # Fallback for older versions or if not available
                    stats["payload_indices"] = "Information not available"
            except Exception as e:
                stats["indices_error"] = str(e)

            logger.info(f"Retrieved stats for collection '{self.collection_name}'")
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"name": self.collection_name, "error": str(e)}

    def get_embedding(self, id: str) -> Optional[List[float]]:
        """
        Get the embedding vector for a specific document ID.

        Args:
            id: Document ID

        Returns:
            Embedding vector or None if not found
        """
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[id],
                with_vectors=True,
            )

            if points and points[0].vector:
                return points[0].vector.get("default")
            return None
        except Exception as e:
            logger.error(f"Error retrieving embedding for ID {id}: {str(e)}")
            return None

    def repair_indices(self) -> Dict[str, Any]:
        """
        Attempt to repair vector store indices.

        Returns:
            Dictionary with repair results
        """
        logger.info(f"Attempting to repair indices for collection '{self.collection_name}'")

        results = {
            "recreated_indices": [],
            "errors": [],
            "success": False
        }

        try:
            # First, get existing indices
            existing_indices = self.client.get_collection_indices(self.collection_name)
            existing_fields = [idx.field_name for idx in existing_indices]

            logger.info(f"Found {len(existing_indices)} existing indices")

            # Delete problematic indices
            for field_name in existing_fields:
                try:
                    logger.info(f"Deleting index for field: {field_name}")
                    self.client.delete_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name
                    )
                    results["recreated_indices"].append(field_name)
                except Exception as e:
                    error_msg = f"Error deleting index for {field_name}: {str(e)}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)

            # Recreate indices
            self._create_payload_indexes()

            # Check if we need to recreate vector indices
            try:
                logger.info("Checking vector indices")
                self.client.update_collection(
                    collection_name=self.collection_name,
                    optimizer_config=rest.OptimizersConfigDiff(
                        indexing_threshold=0  # Force reindexing
                    )
                )
            except Exception as e:
                error_msg = f"Error updating indexing threshold: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

            results["success"] = True
            logger.info("Repair completed successfully")
            return results
        except Exception as e:
            error_msg = f"Error during repair: {str(e)}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            return results