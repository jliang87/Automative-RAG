#!/usr/bin/env python3
"""
Direct Qdrant health check - bypassing our abstractions
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from src.config.settings import settings


def main():
    print("=== Direct Qdrant Health Check ===")

    # Connect directly to Qdrant
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )

    print(f"Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")

    try:
        # 1. List all collections
        collections = client.get_collections()
        print(f"\n1. Collections found: {len(collections.collections)}")
        for collection in collections.collections:
            print(f"   - {collection.name}")

        # 2. Get specific collection info
        collection_name = settings.qdrant_collection
        print(f"\n2. Checking collection: {collection_name}")

        try:
            collection_info = client.get_collection(collection_name)
            print(f"   Status: {collection_info.status}")
            print(f"   Vectors count: {collection_info.vectors_count}")
            print(f"   Points count: {collection_info.points_count}")
            print(f"   Optimizer status: {collection_info.optimizer_status}")

            # 3. Try to count points directly
            count_result = client.count(collection_name)
            print(f"   Direct count: {count_result.count}")

        except Exception as e:
            print(f"   Error getting collection info: {e}")
            return

        # 4. Sample some points
        print(f"\n3. Sampling points from {collection_name}:")
        try:
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=3,
                with_payload=True,
                with_vectors=False
            )

            points = scroll_result[0]
            print(f"   Retrieved {len(points)} sample points")

            for i, point in enumerate(points):
                metadata = point.payload.get("metadata", {})
                content = point.payload.get("page_content", "")
                print(f"\n   Point {i + 1}:")
                print(f"     ID: {point.id}")
                print(f"     Content length: {len(content)}")
                print(f"     Content preview: {content[:100]}...")
                print(f"     Job ID: {metadata.get('job_id', 'N/A')}")
                print(f"     Source: {metadata.get('source', 'N/A')}")
                print(f"     Title: {metadata.get('title', 'N/A')}")

        except Exception as e:
            print(f"   Error sampling points: {e}")

        # 5. Search for your specific job
        job_id = "2bfedf7b-2de3-4c69-aeac-f13cfdca606e"
        print(f"\n4. Searching for job {job_id}:")

        try:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue

            filter_obj = Filter(
                must=[
                    FieldCondition(
                        key="metadata.job_id",
                        match=MatchValue(value=job_id)
                    )
                ]
            )

            search_result = client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_obj,
                limit=10,
                with_payload=True,
                with_vectors=False
            )

            job_points = search_result[0]
            print(f"   Found {len(job_points)} points for job {job_id}")

            for point in job_points:
                content = point.payload.get("page_content", "")
                metadata = point.payload.get("metadata", {})
                print(f"     Point ID: {point.id}")
                print(f"     Content: {content}")
                print(f"     Metadata: {metadata}")

        except Exception as e:
            print(f"   Error searching for job: {e}")

    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")


if __name__ == "__main__":
    main()