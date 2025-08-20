"""
SystemService - System information and capabilities
Handles system status, configurations, and debug operations
Called by WorkflowController for system-related endpoints
"""

import logging
from typing import List, Dict, Any, Optional

from src.models import QueryMode, QueryModeConfig, SystemCapabilities

logger = logging.getLogger(__name__)


class SystemService:
    """
    Service for system information and capabilities
    Provides system status, configurations, and debug functionality
    """

    def __init__(self, vector_store, job_tracker, workflow_model=None):
        self.vector_store = vector_store
        self.job_tracker = job_tracker
        self.workflow_model = workflow_model

    async def get_query_modes(self) -> List[QueryModeConfig]:
        """Get available query modes configuration"""

        modes = [
            QueryModeConfig(
                mode=QueryMode.FACTS,
                name="车辆规格查询",
                description="验证具体的车辆规格参数",
                icon="📌",
                is_default=True,
                use_case="查询确切的技术规格、配置信息",
                two_layer=False,
                examples=[
                    "2023年宝马X5的后备箱容积是多少？",
                    "特斯拉Model 3的充电速度参数",
                    "奔驰E级使用什么轮胎型号？"
                ]
            ),
            QueryModeConfig(
                mode=QueryMode.FEATURES,
                name="新功能建议",
                description="评估是否应该添加某项功能",
                icon="💡",
                is_default=False,
                use_case="产品决策，功能规划",
                two_layer=True,
                examples=[
                    "是否应该为电动车增加氛围灯功能？",
                    "增加模拟引擎声音对用户体验的影响",
                    "AR抬头显示器值得投资吗？"
                ]
            ),
            QueryModeConfig(
                mode=QueryMode.TRADEOFFS,
                name="权衡利弊分析",
                description="分析设计选择的优缺点",
                icon="🧾",
                is_default=False,
                use_case="设计决策，技术选型",
                two_layer=True,
                examples=[
                    "使用模拟声音 vs 自然静音的利弊",
                    "移除物理按键的优缺点分析",
                    "大屏幕 vs 传统仪表盘的对比"
                ]
            ),
            QueryModeConfig(
                mode=QueryMode.SCENARIOS,
                name="用户场景分析",
                description="评估功能在实际使用场景中的表现",
                icon="🧩",
                is_default=False,
                use_case="用户体验设计，产品规划",
                two_layer=True,
                examples=[
                    "长途旅行时这个功能如何表现？",
                    "家庭用户在日常通勤中的体验如何？",
                    "寒冷气候下的性能表现分析"
                ]
            ),
            QueryModeConfig(
                mode=QueryMode.DEBATE,
                name="多角色讨论",
                description="模拟不同角色的观点和讨论",
                icon="🗣️",
                is_default=False,
                use_case="决策支持，全面评估",
                two_layer=False,
                examples=[
                    "产品经理、工程师和用户代表如何看待自动驾驶功能？",
                    "不同团队对电池技术路线的观点",
                    "关于车内空间设计的多方讨论"
                ]
            ),
            QueryModeConfig(
                mode=QueryMode.QUOTES,
                name="原始用户评论",
                description="提取相关的用户评论和反馈",
                icon="🔍",
                is_default=False,
                use_case="市场研究，用户洞察",
                two_layer=False,
                examples=[
                    "用户对续航里程的真实评价",
                    "关于内饰质量的用户反馈",
                    "充电体验的用户评论摘录"
                ]
            )
        ]

        return modes

    async def get_query_mode(self, mode: str) -> Optional[QueryModeConfig]:
        """Get specific query mode configuration"""

        modes = await self.get_query_modes()
        for mode_config in modes:
            if mode_config.mode.value == mode:
                return mode_config
        return None

    async def get_system_capabilities(self) -> SystemCapabilities:
        """Get comprehensive system capabilities"""

        # Get vector store status
        try:
            vector_stats = self.vector_store.get_stats()
            vector_status = "connected"
            vector_health = "healthy"
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            vector_stats = {"error": str(e)}
            vector_status = "error"
            vector_health = "unhealthy"

        # Get job queue status
        try:
            job_stats = self.job_tracker.count_jobs_by_status()
            queue_status = "active"
            queue_health = "healthy"
        except Exception as e:
            logger.error(f"Error getting job stats: {str(e)}")
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
            logger.error(f"Redis health check failed: {str(e)}")
            redis_status = "error"
            redis_health = "unhealthy"

        # Get workflow system status
        workflow_status = "available"
        workflow_health = "healthy"
        if self.workflow_model:
            try:
                workflow_types = self.workflow_model.get_available_workflow_types()
                workflow_status = "active"
            except Exception as e:
                logger.error(f"Error getting workflow status: {str(e)}")
                workflow_status = "error"
                workflow_health = "degraded"

        # Overall system health
        component_statuses = [vector_health, queue_health, redis_health, workflow_health]
        if "unhealthy" in component_statuses:
            system_health = "unhealthy"
        elif "degraded" in component_statuses:
            system_health = "degraded"
        else:
            system_health = "healthy"

        return SystemCapabilities(
            available_modes=[mode.value for mode in QueryMode],
            validation_enabled=False,  # Validation will be removed
            max_query_length=1000,
            supported_languages=["zh", "en"],
            version="2.0.0",  # Updated version for new architecture
            vector_store_status=vector_status,
            redis_status=redis_status,
            job_queue_status=queue_status,
            system_health=system_health,
            component_health={
                "vector_store": vector_health,
                "redis": redis_health,
                "job_queue": queue_health,
                "workflow_system": workflow_health
            },
            system_stats={
                "vector_store": vector_stats,
                "job_queue": job_stats,
                "workflow_system": {
                    "status": workflow_status,
                    "available_types": len(workflow_types) if self.workflow_model else 0
                }
            },
            features={
                "query_modes": len(QueryMode),
                "workflow_system": True,
                "background_processing": True,
                "document_retrieval": True,
                "multi_language": True,
                "video_processing": True,
                "document_processing": True,
                "causation_analysis": False  # Future feature
            }
        )

    async def get_manufacturers(self) -> List[str]:
        """Get available manufacturers from the system"""

        try:
            # Try to extract from vector store metadata
            manufacturers = await self._extract_unique_metadata_values("manufacturer")

            if not manufacturers:
                # Fallback to common manufacturers
                manufacturers = [
                    "BMW", "Mercedes-Benz", "Tesla", "Audi", "Volkswagen",
                    "Toyota", "Honda", "Ford", "Chevrolet", "Nissan",
                    "Hyundai", "Kia", "Volvo", "Lexus", "Acura",
                    "Porsche", "Jaguar", "Land Rover", "Mini", "Smart"
                ]

            return sorted(manufacturers)

        except Exception as e:
            logger.error(f"Error getting manufacturers: {str(e)}")
            return ["BMW", "Tesla", "Mercedes-Benz", "Audi", "Toyota"]

    async def get_models(self, manufacturer: Optional[str] = None) -> List[str]:
        """Get available models, optionally filtered by manufacturer"""

        try:
            if manufacturer:
                models = await self._extract_unique_metadata_values("model", {"manufacturer": manufacturer})
                if not models:
                    models = self._get_fallback_models(manufacturer)
            else:
                models = await self._extract_unique_metadata_values("model")
                if not models:
                    models = [
                        "Model 3", "Model S", "Model Y", "X5", "X3", "3 Series", "5 Series",
                        "E-Class", "C-Class", "S-Class", "A4", "A6", "Q5", "Q7",
                        "Camry", "Accord", "Civic", "Corolla", "F-150", "Mustang"
                    ]

            return sorted(models)

        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return self._get_fallback_models(manufacturer) if manufacturer else ["Model A", "Model B"]

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get comprehensive queue system status"""

        try:
            # Get job statistics
            job_stats = self.job_tracker.count_jobs_by_status()

            # Get detailed queue status
            try:
                from src.core.orchestration.queue_manager import queue_manager
                queue_details = queue_manager.get_queue_status()
                queue_statistics = queue_manager.get_queue_statistics()
            except Exception as e:
                logger.warning(f"Could not get detailed queue status: {str(e)}")
                queue_details = {"error": "Queue details unavailable"}
                queue_statistics = {"error": "Queue statistics unavailable"}

            # Calculate health metrics
            total_jobs = job_stats.get("total", 0)
            failed_jobs = job_stats.get("failed", 0)

            if total_jobs == 0:
                queue_health = "idle"
                success_rate = 100.0
            else:
                failure_rate = failed_jobs / total_jobs
                if failure_rate > 0.1:  # More than 10% failure rate
                    queue_health = "degraded"
                else:
                    queue_health = "healthy"
                success_rate = ((total_jobs - failed_jobs) / total_jobs) * 100

            # Get workflow system status if available
            workflow_queue_status = {}
            if self.workflow_model:
                try:
                    workflow_queue_status = {
                        "workflow_system_active": True,
                        "available_workflow_types": len(self.workflow_model.get_available_workflow_types())
                    }
                except Exception as e:
                    workflow_queue_status = {"workflow_system_error": str(e)}

            return {
                "job_statistics": job_stats,
                "queue_details": queue_details,
                "queue_statistics": queue_statistics,
                "workflow_queues": workflow_queue_status,
                "queue_health": queue_health,
                "success_rate": success_rate,
                "system_health": "healthy" if queue_health != "degraded" else "degraded",
                "performance_metrics": {
                    "average_processing_time": self._calculate_avg_processing_time(),
                    "throughput_per_hour": self._calculate_throughput(),
                    "current_load": self._calculate_current_load(job_stats)
                },
                "last_updated": "now"
            }

        except Exception as e:
            logger.error(f"Error getting queue status: {str(e)}")
            return {
                "error": str(e),
                "queue_health": "error",
                "system_health": "error"
            }

    async def debug_document_retrieval(self, query: str, limit: int = 100) -> Dict[str, Any]:
        """Debug document retrieval from vector store"""

        try:
            import time
            start_time = time.time()

            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=min(limit, 100),
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

            # Calculate statistics
            debug_stats = {
                "query_length": len(query),
                "results_found": len(results),
                "avg_score": sum(float(score) for _, score in results) / len(results) if results else 0,
                "avg_content_length": sum(len(doc.page_content) for doc, _ in results) / len(results) if results else 0,
                "unique_sources": len(set(doc.metadata.get("source", "unknown") for doc, _ in results)),
                "search_performance": {
                    "search_time_ms": round(search_time * 1000, 2),
                    "docs_per_second": round(len(results) / max(search_time, 0.001), 2)
                },
                "quality_distribution": self._analyze_result_quality(results)
            }

            return {
                "query": query,
                "limit_requested": limit,
                "documents_found": len(results),
                "search_time_seconds": round(search_time, 3),
                "documents": debug_documents,
                "debug_statistics": debug_stats,
                "vector_store_info": await self._get_vector_store_info(),
                "recommendations": self._generate_debug_recommendations(debug_stats),
                "system_status": await self._get_retrieval_system_status()
            }

        except Exception as e:
            logger.error(f"Debug retrieval failed: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "documents_found": 0,
                "system_status": "error"
            }

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _extract_unique_metadata_values(self, field: str,
                                            filter_dict: Optional[Dict] = None) -> List[str]:
        """Extract unique values for a metadata field"""

        try:
            # This would typically query the vector store for unique metadata values
            # For now, return empty to trigger fallback
            return []
        except Exception as e:
            logger.error(f"Error extracting metadata values for {field}: {str(e)}")
            return []

    def _get_fallback_models(self, manufacturer: str) -> List[str]:
        """Get fallback models for known manufacturers"""

        fallback_models = {
            "BMW": ["X5", "X3", "3 Series", "5 Series", "7 Series", "i4", "iX", "X1", "X7", "M3", "M5"],
            "Tesla": ["Model 3", "Model S", "Model X", "Model Y", "Cybertruck", "Roadster"],
            "Mercedes-Benz": ["E-Class", "C-Class", "S-Class", "GLE", "GLC", "EQS", "A-Class", "CLA", "G-Class"],
            "Audi": ["A4", "A6", "A8", "Q5", "Q7", "e-tron", "A3", "Q3", "RS6", "TT"],
            "Toyota": ["Camry", "Corolla", "RAV4", "Highlander", "Prius", "Sienna", "Tacoma", "4Runner"],
            "Honda": ["Accord", "Civic", "CR-V", "Pilot", "Odyssey", "HR-V", "Passport", "Ridgeline"],
            "Ford": ["F-150", "Mustang", "Explorer", "Escape", "Edge", "Bronco", "Ranger", "Expedition"],
            "Volkswagen": ["Jetta", "Passat", "Tiguan", "Atlas", "Golf", "ID.4", "Arteon", "Taos"],
            "Hyundai": ["Elantra", "Sonata", "Tucson", "Santa Fe", "Palisade", "Ioniq", "Kona", "Veloster"],
            "Nissan": ["Altima", "Sentra", "Rogue", "Pathfinder", "Leaf", "370Z", "Titan", "Armada"]
        }

        return fallback_models.get(manufacturer, ["Model A", "Model B", "Model C"])

    def _calculate_avg_processing_time(self) -> float:
        """Calculate average processing time for jobs"""

        try:
            # This would analyze completed jobs in a real implementation
            return 25.5  # Placeholder
        except Exception:
            return 0.0

    def _calculate_throughput(self) -> int:
        """Calculate jobs processed per hour"""

        try:
            # This would analyze job completion rates
            return 50  # Placeholder
        except Exception:
            return 0

    def _calculate_current_load(self, job_stats: Dict[str, int]) -> float:
        """Calculate current system load percentage"""

        try:
            active_jobs = job_stats.get("processing", 0) + job_stats.get("pending", 0)
            max_concurrent = 100  # Would come from system configuration
            return (active_jobs / max_concurrent) * 100 if max_concurrent > 0 else 0.0
        except Exception:
            return 0.0

    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content"""

        if not content:
            return "empty"

        content_lower = content.lower()

        if any(word in content_lower for word in ["规格", "参数", "尺寸", "重量", "specifications"]):
            return "specifications"
        elif any(word in content_lower for word in ["评价", "评论", "反馈", "review", "feedback"]):
            return "reviews"
        elif any(word in content_lower for word in ["功能", "特性", "配置", "features", "configuration"]):
            return "features"
        elif any(word in content_lower for word in ["问题", "故障", "维修", "issue", "problem", "repair"]):
            return "issues"
        elif any(word in content_lower for word in ["对比", "比较", "vs", "comparison"]):
            return "comparison"
        else:
            return "general"

    def _assess_metadata_completeness(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Assess completeness of document metadata"""

        expected_fields = ["source", "manufacturer", "model", "year", "category", "title"]
        present_fields = sum(1 for field in expected_fields if metadata.get(field))
        completeness_score = present_fields / len(expected_fields)

        return {
            "score": completeness_score,
            "present_fields": present_fields,
            "total_expected": len(expected_fields),
            "missing_fields": [field for field in expected_fields if not metadata.get(field)],
            "quality_level": "high" if completeness_score > 0.7 else "medium" if completeness_score > 0.4 else "low"
        }

    def _analyze_result_quality(self, results) -> Dict[str, int]:
        """Analyze quality distribution of search results"""

        if not results:
            return {"high": 0, "medium": 0, "low": 0}

        high_quality = sum(1 for _, score in results if score > 0.8)
        medium_quality = sum(1 for _, score in results if 0.5 <= score <= 0.8)
        low_quality = sum(1 for _, score in results if score < 0.5)

        return {
            "high": high_quality,
            "medium": medium_quality,
            "low": low_quality
        }

    async def _get_vector_store_info(self) -> Dict[str, Any]:
        """Get vector store information"""

        try:
            stats = self.vector_store.get_stats()
            return {
                "status": "connected",
                "stats": stats,
                "type": "vector_store"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "type": "vector_store"
            }

    def _generate_debug_recommendations(self, debug_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on debug statistics"""

        recommendations = []

        results_found = debug_stats.get("results_found", 0)
        avg_score = debug_stats.get("avg_score", 0)
        unique_sources = debug_stats.get("unique_sources", 0)
        search_time_ms = debug_stats.get("search_performance", {}).get("search_time_ms", 0)

        if results_found == 0:
            recommendations.append("No documents found - consider broadening the query or checking data availability")
        elif results_found < 5:
            recommendations.append("Few results found - query might be too specific or index needs more content")

        if avg_score < 0.5:
            recommendations.append("Low average relevance scores - consider improving query specificity or document quality")
        elif avg_score > 0.9:
            recommendations.append("Excellent relevance scores - query is well-matched to available content")

        if unique_sources < 2:
            recommendations.append("Results from limited sources - consider expanding data collection or improving source diversity")

        if search_time_ms > 1000:
            recommendations.append("Slow search performance - consider optimizing vector store indices or reducing search scope")
        elif search_time_ms < 100:
            recommendations.append("Excellent search performance - system is well-optimized")

        if not recommendations:
            recommendations.append("Search performance and results quality are within acceptable ranges")

        return recommendations

    async def _get_retrieval_system_status(self) -> Dict[str, Any]:
        """Get retrieval system status"""

        return {
            "vector_store": "connected" if hasattr(self, 'vector_store') else "disconnected",
            "job_tracker": "active" if hasattr(self, 'job_tracker') else "inactive",
            "workflow_system": "active" if self.workflow_model else "inactive",
            "last_health_check": "now"
        }

    def get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics"""

        try:
            job_stats = self.job_tracker.count_jobs_by_status()

            return {
                "system_uptime": "99.9%",  # Placeholder
                "total_requests_processed": job_stats.get("total", 0),
                "success_rate": self._calculate_success_rate(job_stats),
                "average_response_time": self._calculate_avg_processing_time(),
                "current_load": self._calculate_current_load(job_stats),
                "component_health": {
                    "vector_store": "healthy",
                    "job_queue": "healthy",
                    "workflow_system": "healthy",
                    "redis": "healthy"
                },
                "resource_usage": {
                    "memory_usage": "65%",  # Placeholder
                    "cpu_usage": "45%",     # Placeholder
                    "disk_usage": "30%"     # Placeholder
                },
                "feature_usage": {
                    "query_processing": job_stats.get("completed", 0),
                    "document_processing": 0,  # Would track separately
                    "video_processing": 0      # Would track separately
                }
            }

        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return {"error": str(e)}

    def _calculate_success_rate(self, job_stats: Dict[str, int]) -> float:
        """Calculate job success rate percentage"""

        try:
            total = job_stats.get("total", 0)
            completed = job_stats.get("completed", 0)

            if total == 0:
                return 100.0

            return (completed / total) * 100.0

        except Exception:
            return 0.0