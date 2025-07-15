"""
Validation Cache Manager
Manages caching of validation results for smart retry functionality
"""

import json
import hashlib
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ValidationCacheManager:
    """
    Manages validation result caching for smart retry functionality
    """

    def __init__(self):
        from src.core.background.common import get_redis_client
        self.redis = get_redis_client()
        self.cache_prefix = "validation_cache:"
        self.default_expiry_hours = 24

    async def cache_validation_result(self,
                                      query_id: str,
                                      validation_step: str,
                                      result: Dict[str, Any],
                                      documents_hash: str) -> None:
        """Cache validation result for smart retry."""

        try:
            cache_key = self._generate_cache_key(query_id, validation_step, documents_hash)

            cache_data = {
                "query_id": query_id,
                "validation_step": validation_step,
                "result": result,
                "documents_hash": documents_hash,
                "cached_at": datetime.now().isoformat(),
                "cache_version": "v1.0"
            }

            # Cache with expiry
            cache_json = json.dumps(cache_data, ensure_ascii=False)
            expiry_seconds = self.default_expiry_hours * 3600

            self.redis.setex(cache_key, expiry_seconds, cache_json)

            logger.info(f"Cached validation result for {validation_step} in query {query_id}")

        except Exception as e:
            logger.error(f"Error caching validation result: {str(e)}")

    async def get_cached_validation_result(self,
                                           query_id: str,
                                           validation_step: str,
                                           documents_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached validation result if available."""

        try:
            cache_key = self._generate_cache_key(query_id, validation_step, documents_hash)
            cached_data = self.redis.get(cache_key)

            if cached_data:
                cache_data = json.loads(cached_data)

                # Verify cache validity
                if self._is_cache_valid(cache_data):
                    logger.info(f"Found cached validation result for {validation_step}")
                    return cache_data["result"]
                else:
                    # Remove invalid cache
                    self.redis.delete(cache_key)
                    logger.info(f"Removed invalid cache for {validation_step}")

            return None

        except Exception as e:
            logger.error(f"Error retrieving cached validation result: {str(e)}")
            return None

    def _generate_cache_key(self, query_id: str, validation_step: str, documents_hash: str) -> str:
        """Generate cache key for validation result."""

        # Create unique key based on query, step, and documents
        key_components = f"{query_id}:{validation_step}:{documents_hash}"
        key_hash = hashlib.md5(key_components.encode()).hexdigest()

        return f"{self.cache_prefix}{key_hash}"

    def _is_cache_valid(self, cache_data: Dict[str, Any]) -> bool:
        """Check if cached data is still valid."""

        try:
            cached_at_str = cache_data.get("cached_at")
            if not cached_at_str:
                return False

            cached_at = datetime.fromisoformat(cached_at_str)
            expiry_time = cached_at + timedelta(hours=self.default_expiry_hours)

            return datetime.now() < expiry_time

        except Exception:
            return False

    async def invalidate_cache(self, query_id: str) -> int:
        """Invalidate all cached results for a query."""

        try:
            # Find all cache keys for this query
            pattern = f"{self.cache_prefix}*{query_id}*"
            cache_keys = self.redis.keys(pattern)

            if cache_keys:
                deleted_count = self.redis.delete(*cache_keys)
                logger.info(f"Invalidated {deleted_count} cache entries for query {query_id}")
                return deleted_count

            return 0

        except Exception as e:
            logger.error(f"Error invalidating cache for query {query_id}: {str(e)}")
            return 0

    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache usage statistics."""

        try:
            # Count cache entries
            pattern = f"{self.cache_prefix}*"
            cache_keys = self.redis.keys(pattern)

            total_entries = len(cache_keys)

            # Sample some entries to check validity
            valid_entries = 0
            expired_entries = 0

            for key in cache_keys[:100]:  # Sample first 100
                try:
                    cached_data = self.redis.get(key)
                    if cached_data:
                        cache_data = json.loads(cached_data)
                        if self._is_cache_valid(cache_data):
                            valid_entries += 1
                        else:
                            expired_entries += 1
                except Exception:
                    expired_entries += 1

            return {
                "total_cache_entries": total_entries,
                "sampled_entries": min(100, total_entries),
                "valid_entries": valid_entries,
                "expired_entries": expired_entries,
                "cache_hit_rate": "unknown",  # Would track this with usage metrics
                "total_cache_size": f"{len(cache_keys)} keys"
            }

        except Exception as e:
            logger.error(f"Error getting cache statistics: {str(e)}")
            return {"error": str(e)}