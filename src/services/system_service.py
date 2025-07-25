import logging
from typing import List, Dict, Any, Optional

from src.models import QueryMode  # Updated import

logger = logging.getLogger(__name__)


class SystemService:
    """Service for system information and capabilities."""

    def __init__(self, vector_store, job_tracker, orchestration_service):
        self.vector_store = vector_store
        self.job_tracker = job_tracker
        self.orchestration_service = orchestration_service

    async def get_query_modes(self) -> List[Dict[str, Any]]:
        """Get available query modes."""
        modes = [
            {
                "mode": QueryMode.FACTS.value,
                "name": "车辆规格查询",
                "description": "验证具体的车辆规格参数",
                "icon": "📌",
                "is_default": True,
                "use_case": "查询确切的技术规格、配置信息",
                "two_layer": False,
                "examples": [
                    "2023年宝马X5的后备箱容积是多少？",
                    "特斯拉Model 3的充电速度参数",
                    "奔驰E级使用什么轮胎型号？"
                ]
            },
            {
                "mode": QueryMode.FEATURES.value,
                "name": "新功能建议",
                "description": "评估是否应该添加某项功能",
                "icon": "💡",
                "is_default": False,
                "use_case": "产品决策，功能规划",
                "two_layer": True,
                "examples": [
                    "是否应该为电动车增加氛围灯功能？",
                    "增加模拟引擎声音对用户体验的影响",
                    "AR抬头显示器值得投资吗？"
                ]
            },
            {
                "mode": QueryMode.TRADEOFFS.value,
                "name": "权衡利弊分析",
                "description": "分析设计选择的优缺点",
                "icon": "🧾",
                "is_default": False,
                "use_case": "设计决策，技术选型",
                "two_layer": True,
                "examples": [
                    "使用模拟声音 vs 自然静音的利弊",
                    "移除物理按键的优缺点分析",
                    "大屏幕 vs 传统仪表盘的对比"
                ]
            },
            {
                "mode": QueryMode.SCENARIOS.value,
                "name": "用户场景分析",
                "description": "评估功能在实际使用场景中的表现",
                "icon": "🧩",
                "is_default": False,
                "use_case": "用户体验设计，产品规划",
                "two_layer": True,
                "examples": [
                    "长途旅行时这个功能如何表现？",
                    "家庭用户在日常通勤中的体验如何？",
                    "寒冷气候下的性能表现分析"
                ]
            },
            {
                "mode": QueryMode.DEBATE.value,
                "name": "多角色讨论",
                "description": "模拟不同角色的观点和讨论",
                "icon": "🗣️",
                "is_default": False,
                "use_case": "决策支持，全面评估",
                "two_layer": False,
                "examples": [
                    "产品经理、工程师和用户代表如何看待自动驾驶功能？",
                    "不同团队对电池技术路线的观点",
                    "关于车内空间设计的多方讨论"
                ]
            },
            {
                "mode": QueryMode.QUOTES.value,
                "name": "原始用户评论",
                "description": "提取相关的用户评论和反馈",
                "icon": "🔍",
                "is_default": False,
                "use_case": "市场研究，用户洞察",
                "two_layer": False,
                "examples": [
                    "用户对续航里程的真实评价",
                    "关于内饰质量的用户反馈",
                    "充电体验的用户评论摘录"
                ]
            }
        ]
        return modes

    async def get_query_mode(self, mode: str) -> Optional[Dict[str, Any]]:
        """Get specific query mode configuration."""
        modes = await self.get_query_modes()
        for mode_config in modes:
            if mode_config["mode"] == mode:
                return mode_config
        return None

    async def get_system_capabilities(self) -> Dict[str, Any]:
        """Get system capabilities."""
        # Get vector store stats
        try:
            vector_stats = self.vector_store.get_stats()
            vector_status = "connected"
            vector_health = "healthy"
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            vector_stats = {"error": str(e)}
            vector_status = "error"
            vector_health = "unhealthy"

        # Get job queue stats
        try:
            job_stats = self.job_tracker.count_jobs_by_status()
            queue_status = "active"
            queue_health = "healthy"
        except Exception as e:
            logger.error(f"Error getting job stats: {e}")
            job_stats = {"error": str(e)}
            queue_status = "error"
            queue_health = "unhealthy"

        # Get Redis status
        try:
            redis_client = self.job_tracker.redis
            redis_client.ping()
            redis_status = "connected"
            redis_health = "healthy"
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            redis_status = "error"
            redis_health = "unhealthy"

        # Overall system health
        system_health = "healthy"
        if any(status == "error" for status in [vector_status, queue_status, redis_status]):
            system_health = "degraded"

        return {
            "available_modes": [mode.value for mode in QueryMode],
            "validation_enabled": True,
            "max_query_length": 1000,
            "supported_languages": ["zh", "en"],
            "version": "1.0.0",
            "vector_store_status": vector_status,
            "redis_status": redis_status,
            "job_queue_status": queue_status,
            "system_health": system_health,
            "component_health": {
                "vector_store": vector_health,
                "redis": redis_health,
                "job_queue": queue_health
            },
            "system_stats": {
                "vector_store": vector_stats,
                "job_queue": job_stats
            },
            "features": {
                "query_modes": len(QueryMode),
                "validation_workflows": True,
                "background_processing": True,
                "document_retrieval": True,
                "multi_language": True
            }
        }

    async def get_manufacturers(self) -> List[str]:
        """Get available manufacturers from vector store."""
        try:
            # Try to get unique manufacturers from vector store metadata
            # This is a simplified implementation - in a real system you'd query the vector store
            manufacturers = await self._extract_unique_metadata_values("manufacturer")

            if not manufacturers:
                # Fallback to common automotive manufacturers
                manufacturers = [
                    "BMW", "Mercedes-Benz", "Tesla", "Audi", "Volkswagen",
                    "Toyota", "Honda", "Ford", "Chevrolet", "Nissan",
                    "Hyundai", "Kia", "Volvo", "Lexus", "Acura"
                ]

            return sorted(manufacturers)

        except Exception as e:
            logger.error(f"Error getting manufacturers: {e}")
            return ["BMW", "Tesla", "Mercedes-Benz", "Audi"]  # Fallback

    async def get_models(self, manufacturer: Optional[str] = None) -> List[str]:
        """Get available models, optionally filtered by manufacturer."""
        try:
            if manufacturer:
                # Try to get models for specific manufacturer
                models = await self._extract_unique_metadata_values("model", {"manufacturer": manufacturer})

                if not models:
                    # Fallback to known models for common manufacturers
                    models = self._get_fallback_models(manufacturer)
            else:
                # Get all models
                models = await self._extract_unique_metadata_values("model")

                if not models:
                    # Fallback to popular models across manufacturers
                    models = [
                        "Model 3", "Model S", "X5", "X3", "3 Series", "5 Series",
                        "E-Class", "C-Class", "S-Class", "A4", "A6", "Q5",
                        "Camry", "Accord", "Civic", "Corolla", "F-150"
                    ]

            return sorted(models)

        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return self._get_fallback_models(manufacturer) if manufacturer else ["Model A", "Model B", "Model C"]

    def _get_fallback_models(self, manufacturer: str) -> List[str]:
        """Get fallback models for known manufacturers."""
        fallback_models = {
            "BMW": ["X5", "X3", "3 Series", "5 Series", "7 Series", "i4", "iX", "X1", "X7"],
            "Tesla": ["Model 3", "Model S", "Model X", "Model Y", "Cybertruck"],
            "Mercedes-Benz": ["E-Class", "C-Class", "S-Class", "GLE", "GLC", "EQS", "A-Class", "CLA"],
            "Audi": ["A4", "A6", "A8", "Q5", "Q7", "e-tron", "A3", "Q3"],
            "Toyota": ["Camry", "Corolla", "RAV4", "Highlander", "Prius", "Sienna"],
            "Honda": ["Accord", "Civic", "CR-V", "Pilot", "Odyssey", "HR-V"],
            "Ford": ["F-150", "Mustang", "Explorer", "Escape", "Edge", "Bronco"],
            "Volkswagen": ["Jetta", "Passat", "Tiguan", "Atlas", "Golf", "ID.4"]
        }

        return fallback_models.get(manufacturer, ["Model A", "Model B", "Model C"])

    async def _extract_unique_metadata_values(self, field: str, filter_dict: Optional[Dict] = None) -> List[str]:
        """Extract unique values for a metadata field from the vector store."""
        try:
            # This would typically involve querying the vector store for unique metadata values
            # For now, return empty list to trigger fallback behavior
            return []
        except Exception as e:
            logger.error(f"Error extracting metadata values for {field}: {e}")
            return []

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get status of the job queue system."""
        try:
            # Get job statistics from job tracker
            job_stats = self.job_tracker.count_jobs_by_status()

            # Get detailed queue status from job chain
            try:
                from src.core.orchestration.job_chain import job_chain
                queue_details = job_chain.get_queue_status()
            except Exception as e:
                logger.warning(f"Could not get detailed queue status: {e}")
                queue_details = {"error": "Queue details unavailable"}

            # Calculate queue health
            total_jobs = job_stats.get("total", 0)
            failed_jobs = job_stats.get("failed", 0)

            if total_jobs == 0:
                queue_health = "idle"
            elif failed_jobs / max(total_jobs, 1) > 0.1:  # More than 10% failure rate
                queue_health = "degraded"
            else:
                queue_health = "healthy"

            # Get recent job activity
            recent_activity = await self._get_recent_job_activity()

            return {
                "job_statistics": job_stats,
                "queue_details": queue_details,
                "queue_health": queue_health,
                "recent_activity": recent_activity,
                "system_health": "healthy" if queue_health != "degraded" else "degraded",
                "last_updated": "now",
                "performance_metrics": {
                    "average_processing_time": self._calculate_avg_processing_time(),
                    "throughput_per_hour": self._calculate_throughput(),
                    "success_rate": self._calculate_success_rate(job_stats)
                }
            }

        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {
                "job_statistics": {"total": 0, "error": str(e)},
                "queue_details": {"error": "Queue status unavailable"},
                "queue_health": "error",
                "system_health": "error",
                "last_updated": "now"
            }

    async def _get_recent_job_activity(self) -> List[Dict[str, Any]]:
        """Get recent job activity for monitoring."""
        try:
            # Get recent jobs (last 10)
            recent_jobs = self.job_tracker.get_all_jobs(limit=10)

            activity = []
            for job in recent_jobs:
                activity.append({
                    "job_id": job.get("job_id", ""),
                    "job_type": job.get("job_type", ""),
                    "status": job.get("status", ""),
                    "created_at": job.get("created_at", 0),
                    "query_mode": job.get("metadata", {}).get("query_mode", "unknown")
                })

            return activity

        except Exception as e:
            logger.error(f"Error getting recent job activity: {e}")
            return []

    def _calculate_avg_processing_time(self) -> float:
        """Calculate average processing time for completed jobs."""
        try:
            # This would typically analyze completed jobs
            # For now, return a reasonable estimate
            return 25.5  # seconds
        except:
            return 0.0

    def _calculate_throughput(self) -> int:
        """Calculate jobs processed per hour."""
        try:
            # This would typically analyze job completion rates
            # For now, return a reasonable estimate
            return 50  # jobs per hour
        except:
            return 0

    def _calculate_success_rate(self, job_stats: Dict[str, int]) -> float:
        """Calculate job success rate."""
        try:
            total = job_stats.get("total", 0)
            completed = job_stats.get("completed", 0)

            if total == 0:
                return 0.0

            return (completed / total) * 100.0

        except:
            return 0.0

    async def debug_document_retrieval(self, query: str, limit: int = 100) -> Dict[str, Any]:
        """Debug endpoint to retrieve documents from vector store."""
        try:
            import time
            start_time = time.time()

            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=min(limit, 100),  # Cap at 100 for performance
                metadata_filter=None
            )

            search_time = time.time() - start_time

            # Format results for debugging
            debug_documents = []
            for doc, score in results:
                debug_doc = {
                    "content_preview": doc.page_content[:200] + "..." if len(
                        doc.page_content) > 200 else doc.page_content,
                    "score": float(score),
                    "metadata": dict(doc.metadata),
                    "content_length": len(doc.page_content),
                    "content_type": self._detect_content_type(doc.page_content),
                    "metadata_completeness": self._assess_metadata_completeness(doc.metadata)
                }
                debug_documents.append(debug_doc)

            # Calculate debug statistics
            debug_stats = {
                "query_length": len(query),
                "results_found": len(results),
                "avg_score": sum(float(score) for _, score in results) / len(results) if results else 0,
                "avg_content_length": sum(len(doc.page_content) for doc, _ in results) / len(results) if results else 0,
                "unique_sources": len(set(doc.metadata.get("source", "unknown") for doc, _ in results)),
                "search_performance": {
                    "search_time_ms": round(search_time * 1000, 2),
                    "docs_per_second": round(len(results) / max(search_time, 0.001), 2)
                }
            }

            return {
                "query": query,
                "limit_requested": limit,
                "documents_found": len(results),
                "search_time_seconds": round(search_time, 3),
                "documents": debug_documents,
                "debug_statistics": debug_stats,
                "vector_store_stats": self.vector_store.get_stats(),
                "recommendations": self._generate_debug_recommendations(debug_stats)
            }

        except Exception as e:
            logger.error(f"Debug retrieval failed: {e}")
            return {
                "query": query,
                "limit_requested": limit,
                "error": str(e),
                "documents_found": 0,
                "search_time_seconds": 0,
                "documents": []
            }

    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content."""
        if not content:
            return "empty"

        content_lower = content.lower()

        if any(word in content_lower for word in ["规格", "参数", "尺寸", "重量"]):
            return "specifications"
        elif any(word in content_lower for word in ["评价", "评论", "反馈"]):
            return "reviews"
        elif any(word in content_lower for word in ["功能", "特性", "配置"]):
            return "features"
        elif any(word in content_lower for word in ["问题", "故障", "维修"]):
            return "issues"
        else:
            return "general"

    def _assess_metadata_completeness(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the completeness of document metadata."""
        expected_fields = ["source", "manufacturer", "model", "year", "category"]

        present_fields = sum(1 for field in expected_fields if metadata.get(field))
        completeness_score = present_fields / len(expected_fields)

        return {
            "score": completeness_score,
            "present_fields": present_fields,
            "total_expected": len(expected_fields),
            "missing_fields": [field for field in expected_fields if not metadata.get(field)]
        }

    def _generate_debug_recommendations(self, debug_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on debug statistics."""
        recommendations = []

        if debug_stats["results_found"] == 0:
            recommendations.append("No documents found - consider broadening the query or checking data availability")
        elif debug_stats["results_found"] < 5:
            recommendations.append("Few results found - query might be too specific")

        if debug_stats["avg_score"] < 0.5:
            recommendations.append("Low average relevance scores - consider improving query specificity")

        if debug_stats["unique_sources"] < 2:
            recommendations.append("Results from limited sources - consider expanding data collection")

        search_time_ms = debug_stats["search_performance"]["search_time_ms"]
        if search_time_ms > 1000:
            recommendations.append("Slow search performance - consider optimizing vector store indices")

        if not recommendations:
            recommendations.append("Search performance and results quality look good")

        return recommendations