#!/usr/bin/env python3
"""
Complete system reset script - clears all data and jobs
"""

import os
import sys
import logging
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import settings
from qdrant_client import QdrantClient
import redis

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clear_qdrant_collection():
    """Delete the Qdrant collection (will be auto-recreated on container restart)"""
    logger.info("🗑️ Clearing Qdrant vector store...")

    try:
        # Connect to Qdrant
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )

        collection_name = settings.qdrant_collection

        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if collection_name in collection_names:
            logger.info(f"Deleting collection: {collection_name}")
            client.delete_collection(collection_name)
            logger.info("✅ Collection deleted successfully")
        else:
            logger.info(f"Collection {collection_name} doesn't exist")

        logger.info("✅ Qdrant vector store cleared")
        logger.info("ℹ️ Collection will be auto-recreated when containers restart")
        return True

    except Exception as e:
        logger.error(f"❌ Error clearing Qdrant: {str(e)}")
        return False


def clear_redis_jobs():
    """Clear all job data from Redis"""
    logger.info("🗑️ Clearing Redis job data...")

    try:
        # Connect to Redis
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_port = int(os.environ.get("REDIS_PORT", "6379"))
        redis_password = os.environ.get("REDIS_PASSWORD", None)

        redis_kwargs = {
            "host": redis_host,
            "port": redis_port,
            "decode_responses": True,
        }
        if redis_password:
            redis_kwargs["password"] = redis_password

        redis_client = redis.Redis(**redis_kwargs)

        # Test connection
        redis_client.ping()
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")

        # Clear different types of job data
        patterns_to_clear = [
            "rag_system:jobs*",  # Main job tracking
            "rag_system:job_progress*",  # Job progress
            "job_chain:*",  # Job chain states
            "queue_busy:*",  # Queue status
            "waiting_tasks:*",  # Waiting tasks
            "dramatiq:*",  # Dramatiq queues and messages
            "model_loaded:*",  # Model loading status
            "model_loading_time:*",  # Model timing
        ]

        total_deleted = 0
        for pattern in patterns_to_clear:
            keys = redis_client.keys(pattern)
            if keys:
                deleted = redis_client.delete(*keys)
                total_deleted += deleted
                logger.info(f"Deleted {deleted} keys matching {pattern}")

        logger.info(f"✅ Redis cleared - deleted {total_deleted} total keys")
        return True

    except Exception as e:
        logger.error(f"❌ Error clearing Redis: {str(e)}")
        return False


def clear_uploaded_files():
    """Clear uploaded files and media"""
    logger.info("🗑️ Clearing uploaded files...")

    try:
        import shutil

        dirs_to_clear = [
            "data/uploads",
            "data/videos",
            "data/youtube",
            "data/bilibili",
            "data/videos/audio",
        ]

        for dir_path in dirs_to_clear:
            if os.path.exists(dir_path):
                logger.info(f"Clearing directory: {dir_path}")
                shutil.rmtree(dir_path)
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"✅ Cleared {dir_path}")
            else:
                logger.info(f"Directory {dir_path} doesn't exist")

        logger.info("✅ Uploaded files cleared")
        return True

    except Exception as e:
        logger.error(f"❌ Error clearing files: {str(e)}")
        return False


def clear_logs():
    """Clear log files"""
    logger.info("🗑️ Clearing log files...")

    try:
        log_files = [
            "logs/verification.log",
            "logs/app.log",
            "logs/worker.log",
        ]

        for log_file in log_files:
            if os.path.exists(log_file):
                os.remove(log_file)
                logger.info(f"Deleted {log_file}")

        logger.info("✅ Log files cleared")
        return True

    except Exception as e:
        logger.error(f"❌ Error clearing logs: {str(e)}")
        return False


def verify_cleanup():
    """Verify that cleanup was successful"""
    logger.info("🔍 Verifying cleanup...")

    issues = []

    # Check Qdrant
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]

        if settings.qdrant_collection in collection_names:
            collection_info = client.get_collection(settings.qdrant_collection)
            count = getattr(collection_info, 'vectors_count', 0) or 0
            if count > 0:
                issues.append(f"Qdrant collection still has {count} documents")
            else:
                logger.info("✅ Qdrant collection is empty")
        else:
            logger.info("✅ Qdrant collection doesn't exist")

    except Exception as e:
        issues.append(f"Cannot verify Qdrant: {str(e)}")

    # Check Redis
    try:
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_port = int(os.environ.get("REDIS_PORT", "6379"))
        redis_password = os.environ.get("REDIS_PASSWORD", None)

        redis_kwargs = {
            "host": redis_host,
            "port": redis_port,
            "decode_responses": True,
        }
        if redis_password:
            redis_kwargs["password"] = redis_password

        redis_client = redis.Redis(**redis_kwargs)

        job_keys = redis_client.keys("rag_system:jobs*")
        if job_keys:
            issues.append(f"Redis still has {len(job_keys)} job keys")
        else:
            logger.info("✅ Redis job data cleared")

    except Exception as e:
        issues.append(f"Cannot verify Redis: {str(e)}")

    if issues:
        logger.warning("⚠️ Cleanup verification found issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    else:
        logger.info("✅ Cleanup verification passed")
        return True


def main():
    print("🧹 COMPLETE SYSTEM RESET")
    print("=" * 50)
    print("This will delete:")
    print("  - All vector store documents")
    print("  - All job data and progress")
    print("  - All uploaded files")
    print("  - All log files")
    print("  - All Redis cache data")
    print("")

    confirm = input("Are you sure you want to proceed? Type 'YES' to confirm: ")

    if confirm != "YES":
        print("❌ Operation cancelled")
        return

    print("\n🚀 Starting cleanup...")

    success_count = 0
    total_operations = 4

    # Clear Qdrant
    if clear_qdrant_collection():
        success_count += 1

    # Clear Redis
    if clear_redis_jobs():
        success_count += 1

    # Clear files
    if clear_uploaded_files():
        success_count += 1

    # Clear logs
    if clear_logs():
        success_count += 1

    print(f"\n📊 Cleanup completed: {success_count}/{total_operations} operations successful")

    # Verify cleanup
    print("\n🔍 Verifying cleanup...")
    if verify_cleanup():
        print("\n🎉 SYSTEM RESET COMPLETE!")
        print("✅ Your system is now clean and ready for fresh data")
        print("\n⚠️ IMPORTANT: RESTART YOUR CONTAINERS NOW")
        print("Run: docker-compose down && docker-compose up -d")
        print("\nNext steps:")
        print("  1. Restart containers (see above)")
        print("  2. Apply the job chain metadata fixes")
        print("  3. Upload new videos through the UI")
        print("  4. Videos will be processed with fixed metadata")
        print("  5. Try queries with proper model names")
    else:
        print("\n⚠️ Cleanup completed but verification found some issues")
        print("You may need to manually check the reported problems")
        print("\n⚠️ IMPORTANT: RESTART YOUR CONTAINERS")
        print("Run: docker-compose down && docker-compose up -d")


if __name__ == "__main__":
    main()