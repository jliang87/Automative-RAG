import uuid
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.models.schema import (
    EnhancedQueryRequest,
    EnhancedQueryResponse,
    EnhancedBackgroundJobResponse,
    QueryMode,
    QueryValidationResult,
    ValidationType,
    DocumentResponse
)
from src.services.validation_service import ValidationService
from src.services.orchestration_service import OrchestrationService
from src.services.document_processing_service import DocumentProcessingService
from src.services.response_processing_service import ResponseProcessingService
from src.core.orchestration.job_tracker import job_tracker
from src.core.orchestration.job_chain import job_chain

logger = logging.getLogger(__name__)


class QueryService:
    """
    Core business logic for query processing.
    Orchestrates the entire query workflow without HTTP concerns.
    """

    def __init__(self,
                 validation_service: ValidationService,
                 orchestration_service: OrchestrationService,
                 document_service: DocumentProcessingService,
                 response_service: ResponseProcessingService):
        self.validation_service = validation_service
        self.orchestration_service = orchestration_service
        self.document_service = document_service
        self.response_service = response_service

        # Query mode configurations (moved from router)
        self.query_mode_configs = self._initialize_query_mode_configs()

    def _initialize_query_mode_configs(self) -> Dict[QueryMode, Dict[str, Any]]:
        """Initialize query mode configurations."""
        return {
            QueryMode.FACTS: {
                "mode": QueryMode.FACTS,
                "icon": "📌",
                "name": "车辆规格查询",
                "description": "验证具体的车辆规格参数",
                "use_case": "查询确切的技术规格、配置信息",
                "two_layer": False,
                "is_default": True,
                "examples": [
                    "2023年宝马X5的后备箱容积是多少？",
                    "特斯拉Model 3的充电速度参数",
                    "奔驰E级使用什么轮胎型号？"
                ]
            },
            QueryMode.FEATURES: {
                "mode": QueryMode.FEATURES,
                "icon": "💡",
                "name": "新功能建议",
                "description": "评估是否应该添加某项功能",
                "use_case": "产品决策，功能规划",
                "two_layer": True,
                "examples": [
                    "是否应该为电动车增加氛围灯功能？",
                    "增加模拟引擎声音对用户体验的影响",
                    "AR抬头显示器值得投资吗？"
                ]
            },
            QueryMode.TRADEOFFS: {
                "mode": QueryMode.TRADEOFFS,
                "icon": "🧾",
                "name": "权衡利弊分析",
                "description": "分析设计选择的优缺点",
                "use_case": "设计决策，技术选型",
                "two_layer": True,
                "examples": [
                    "使用模拟声音 vs 自然静音的利弊",
                    "移除物理按键的优缺点分析",
                    "大屏幕 vs 传统仪表盘的对比"
                ]
            },
            QueryMode.SCENARIOS: {
                "mode": QueryMode.SCENARIOS,
                "icon": "🧩",
                "name": "用户场景分析",
                "description": "评估功能在实际使用场景中的表现",
                "use_case": "用户体验设计，产品规划",
                "two_layer": True,
                "examples": [
                    "长途旅行时这个功能如何表现？",
                    "家庭用户在日常通勤中的体验如何？",
                    "寒冷气候下的性能表现分析"
                ]
            },
            QueryMode.DEBATE: {
                "mode": QueryMode.DEBATE,
                "icon": "🗣️",
                "name": "多角色讨论",
                "description": "模拟不同角色的观点和讨论",
                "use_case": "决策支持，全面评估",
                "two_layer": False,
                "examples": [
                    "产品经理、工程师和用户代表如何看待自动驾驶功能？",
                    "不同团队对电池技术路线的观点",
                    "关于车内空间设计的多方讨论"
                ]
            },
            QueryMode.QUOTES: {
                "mode": QueryMode.QUOTES,
                "icon": "🔍",
                "name": "原始用户评论",
                "description": "提取相关的用户评论和反馈",
                "use_case": "市场研究，用户洞察",
                "two_layer": False,
                "examples": [
                    "用户对续航里程的真实评价",
                    "关于内饰质量的用户反馈",
                    "充电体验的用户评论摘录"
                ]
            }
        }

    async def process_query(self, request: EnhancedQueryRequest) -> EnhancedBackgroundJobResponse:
        """
        Core business logic for processing queries.
        Moved from handle_unified_query_with_validation().
        """
        try:
            job_id = str(uuid.uuid4())

            # Validate query mode
            if request.query_mode not in self.query_mode_configs:
                raise ValueError(f"Unsupported query mode: {request.query_mode}")

            mode_config = self.query_mode_configs[request.query_mode]

            # Handle validation configuration
            validation_workflow = None
            validation_id = None
            validation_enabled = False

            if request.validation_config and request.validation_config.enabled:
                validation_workflow = await self.validation_service.create_workflow(
                    job_id, request.validation_config.dict()
                )
                validation_id = validation_workflow.validation_id
                validation_enabled = True
                logger.info(f"Validation enabled for job {job_id} with workflow {validation_id}")

            # Calculate complexity and processing time
            complexity = self._determine_complexity(request.query_mode)
            estimated_time = self._estimate_processing_time(request.query_mode, validation_enabled, validation_workflow)

            # Create job metadata
            job_metadata = self._create_job_metadata(request, mode_config, complexity, estimated_time,
                                                     validation_enabled, validation_id, validation_workflow)

            # Create job record
            job_tracker.create_job(
                job_id=job_id,
                job_type="llm_inference",
                metadata=job_metadata
            )

            # Start orchestration workflow
            await self.orchestration_service.start_query_workflow(
                job_id=job_id,
                request=request,
                validation_enabled=validation_enabled,
                validation_id=validation_id,
                validation_workflow=validation_workflow
            )

            return EnhancedBackgroundJobResponse(
                message=f"Query processing in '{mode_config['name']}' mode" +
                        (f" with {validation_workflow.validation_type.value} validation" if validation_enabled else ""),
                job_id=job_id,
                job_type="llm_inference",
                query_mode=request.query_mode,
                expected_processing_time=estimated_time,
                status="processing",
                complexity_level=complexity,
                validation_enabled=validation_enabled,
                validation_id=validation_id,
                validation_type=validation_workflow.validation_type if validation_workflow else None
            )

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    async def get_query_result(self, job_id: str) -> Optional[EnhancedQueryResponse]:
        """
        Get query results with integrated validation data.
        Moved from the router's get_query_result endpoint.
        """
        job_data = job_tracker.get_job(job_id)

        if not job_data:
            return None

        status = job_data.get("status", "")
        metadata = job_data.get("metadata", {})
        result = job_data.get("result", {})

        # Parse result if it's a JSON string
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except:
                result = {"answer": result}

        # Get query mode
        query_mode = metadata.get("query_mode", "facts")
        try:
            query_mode_enum = QueryMode(query_mode)
        except ValueError:
            query_mode_enum = QueryMode.FACTS

        # Get validation workflow if it exists
        validation_workflow = None
        validation_summary = None
        validation_id = metadata.get("validation_id")

        if validation_id:
            validation_workflow = await self.validation_service.get_validation_workflow(validation_id)
            if validation_workflow:
                validation_summary = self._create_validation_summary(validation_workflow)

        if status == "completed":
            return await self._create_completed_response(result, metadata, query_mode_enum,
                                                         validation_workflow, validation_summary, job_id)
        elif status == "failed":
            return self._create_failed_response(job_data, query_mode_enum,
                                                validation_workflow, validation_summary, job_id)
        else:
            return self._create_processing_response(job_id, metadata, query_mode_enum,
                                                    validation_workflow, validation_summary)

    async def validate_query_compatibility(self, request: EnhancedQueryRequest) -> QueryValidationResult:
        """
        Validate query for mode compatibility and suggest validation type.
        Moved from validate_query_for_modes endpoint.
        """
        try:
            query_text = request.query.lower()

            # Mode compatibility scoring
            mode_scores = self._calculate_mode_compatibility_scores(query_text)

            # Find best matching mode
            suggested_mode = max(mode_scores.items(), key=lambda x: x[1])[0]
            confidence = max(mode_scores.values())

            # Create compatibility map
            compatibility_scores = {mode: score > 0.2 for mode, score in mode_scores.items()}

            # Determine validation recommendations
            validation_recommended, suggested_validation_type = self._determine_validation_recommendation(
                mode_scores, confidence
            )

            # Generate recommendations
            recommendations = self._generate_compatibility_recommendations(
                confidence, mode_scores, validation_recommended, suggested_validation_type
            )

            return QueryValidationResult(
                is_valid=True,
                mode_compatibility=compatibility_scores,
                recommendations=recommendations,
                suggested_mode=suggested_mode,
                confidence_score=confidence,
                suggested_validation_type=suggested_validation_type,
                validation_recommended=validation_recommended
            )

        except Exception as e:
            logger.error(f"Error validating query: {str(e)}")
            return QueryValidationResult(
                is_valid=False,
                mode_compatibility={},
                recommendations=["查询验证失败，将使用默认的车辆规格查询模式"],
                suggested_mode=QueryMode.FACTS,
                confidence_score=0.0,
                suggested_validation_type=ValidationType.BASIC,
                validation_recommended=True
            )

    def _determine_complexity(self, query_mode: QueryMode) -> str:
        """Determine query complexity based on mode."""
        complexity_map = {
            QueryMode.FACTS: "simple",
            QueryMode.FEATURES: "moderate",
            QueryMode.TRADEOFFS: "complex",
            QueryMode.SCENARIOS: "complex",
            QueryMode.DEBATE: "complex",
            QueryMode.QUOTES: "simple"
        }
        return complexity_map.get(query_mode, "simple")

    def _estimate_processing_time(self, query_mode: QueryMode, validation_enabled: bool,
                                  validation_workflow: Any) -> int:
        """Estimate processing time for the query."""
        base_times = {
            QueryMode.FACTS: 10,
            QueryMode.FEATURES: 30,
            QueryMode.TRADEOFFS: 45,
            QueryMode.SCENARIOS: 40,
            QueryMode.DEBATE: 50,
            QueryMode.QUOTES: 20
        }

        estimated_time = base_times.get(query_mode, 20)

        if validation_enabled and validation_workflow:
            validation_overhead = {
                ValidationType.BASIC: 5,
                ValidationType.COMPREHENSIVE: 15,
                ValidationType.USER_GUIDED: 10
            }
            estimated_time += validation_overhead.get(validation_workflow.validation_type, 5)

        return estimated_time

    def _create_job_metadata(self, request: EnhancedQueryRequest, mode_config: Dict[str, Any],
                             complexity: str, estimated_time: int, validation_enabled: bool,
                             validation_id: Optional[str], validation_workflow: Any) -> Dict[str, Any]:
        """Create comprehensive job metadata."""
        return {
            "query": request.query,
            "metadata_filter": request.metadata_filter,
            "top_k": request.top_k,
            "query_mode": request.query_mode.value,
            "has_custom_template": request.prompt_template is not None,
            "complexity_level": complexity,
            "estimated_time": estimated_time,
            "mode_name": mode_config["name"],
            "validation_enabled": validation_enabled,
            "validation_id": validation_id,
            "validation_type": validation_workflow.validation_type.value if validation_workflow else None,
            "unified_system": True
        }

    def _create_validation_summary(self, validation_workflow: Any) -> Dict[str, Any]:
        """Create validation summary for response."""
        return {
            "validation_id": validation_workflow.validation_id,
            "status": validation_workflow.overall_status.value,
            "type": validation_workflow.validation_type.value,
            "confidence": validation_workflow.overall_confidence,
            "passed": validation_workflow.validation_passed,
            "steps_completed": len([s for s in validation_workflow.steps if s.status.value == "completed"]),
            "total_steps": len(validation_workflow.steps),
            "awaiting_user_input": validation_workflow.awaiting_user_input,
            "current_step": validation_workflow.current_step.value if validation_workflow.current_step else None,
            "issues_identified": len(validation_workflow.issues_identified),
            "recommendations": len(validation_workflow.final_recommendations)
        }

    async def _create_completed_response(self, result: Dict[str, Any], metadata: Dict[str, Any],
                                         query_mode_enum: QueryMode, validation_workflow: Any,
                                         validation_summary: Optional[Dict], job_id: str) -> EnhancedQueryResponse:
        """Create response for completed jobs."""
        documents = result.get("documents", [])
        cleaned_documents = await self.document_service.clean_documents(documents)
        processing_stats = self.document_service.calculate_processing_statistics(documents)

        # Clean answer
        answer = self.response_service.clean_answer(result.get("answer", ""))

        # Parse structured sections for two-layer modes
        analysis_structure = None
        mode_config = self.query_mode_configs.get(query_mode_enum)
        if mode_config and mode_config.get("two_layer"):
            analysis_structure = self.response_service.parse_structured_answer(answer, query_mode_enum.value)

        mode_metadata = {
            "processing_mode": query_mode_enum.value,
            "complexity_level": metadata.get("complexity_level", "simple"),
            "mode_name": metadata.get("mode_name", ""),
            "processing_stats": processing_stats,
            "validation_enabled": metadata.get("validation_enabled", False),
            "unified_system": True
        }

        return EnhancedQueryResponse(
            query=result.get("query", metadata.get("query", "")),
            answer=answer,
            documents=cleaned_documents,
            query_mode=query_mode_enum,
            analysis_structure=analysis_structure,
            metadata_filters_used=metadata.get("metadata_filter"),
            execution_time=result.get("execution_time", 0),
            status="completed",
            job_id=job_id,
            mode_metadata=mode_metadata,
            validation=validation_workflow,
            validation_summary=validation_summary
        )

    def _create_failed_response(self, job_data: Dict[str, Any], query_mode_enum: QueryMode,
                                validation_workflow: Any, validation_summary: Optional[Dict],
                                job_id: str) -> EnhancedQueryResponse:
        """Create response for failed jobs."""
        metadata = job_data.get("metadata", {})
        error_msg = job_data.get('error', 'Unknown error')

        return EnhancedQueryResponse(
            query=metadata.get("query", ""),
            answer=f"Error: {error_msg}",
            documents=[],
            query_mode=query_mode_enum,
            metadata_filters_used=metadata.get("metadata_filter"),
            execution_time=0,
            status="failed",
            job_id=job_id,
            validation=validation_workflow,
            validation_summary=validation_summary
        )

    def _create_processing_response(self, job_id: str, metadata: Dict[str, Any],
                                    query_mode_enum: QueryMode, validation_workflow: Any,
                                    validation_summary: Optional[Dict]) -> EnhancedQueryResponse:
        """Create response for jobs still processing."""
        chain_status = job_chain.get_job_chain_status(job_id)
        progress_msg = "Processing query..."

        if chain_status:
            current_task = chain_status.get("current_task", "unknown")
            mode_name = metadata.get("mode_name", query_mode_enum.value)

            if validation_workflow and validation_workflow.awaiting_user_input:
                progress_msg = f"Awaiting user input for validation..."
            else:
                progress_msg = f"Processing {mode_name}... (Current: {current_task})"

        return EnhancedQueryResponse(
            query=metadata.get("query", ""),
            answer=progress_msg,
            documents=[],
            query_mode=query_mode_enum,
            metadata_filters_used=metadata.get("metadata_filter"),
            execution_time=0,
            status="processing",
            job_id=job_id,
            validation=validation_workflow,
            validation_summary=validation_summary
        )

    def _calculate_mode_compatibility_scores(self, query_text: str) -> Dict[QueryMode, float]:
        """Calculate compatibility scores for each query mode."""
        # Keywords for each mode
        facts_keywords = ["什么", "多少", "哪个", "规格", "参数", "尺寸", "重量"]
        features_keywords = ["应该", "值得", "建议", "增加", "功能", "是否"]
        tradeoffs_keywords = ["vs", "对比", "优缺点", "利弊", "比较", "选择"]
        scenarios_keywords = ["场景", "情况下", "使用时", "体验", "用户", "日常"]
        debate_keywords = ["观点", "看法", "认为", "角度", "团队", "讨论"]
        quotes_keywords = ["评价", "反馈", "评论", "用户说", "真实", "体验"]

        # Calculate scores
        mode_scores = {
            QueryMode.FACTS: max(
                sum(1 for kw in facts_keywords if kw in query_text) / len(facts_keywords),
                0.4  # Default minimum for facts mode
            ),
            QueryMode.FEATURES: sum(1 for kw in features_keywords if kw in query_text) / len(features_keywords),
            QueryMode.TRADEOFFS: sum(1 for kw in tradeoffs_keywords if kw in query_text) / len(tradeoffs_keywords),
            QueryMode.SCENARIOS: sum(1 for kw in scenarios_keywords if kw in query_text) / len(scenarios_keywords),
            QueryMode.DEBATE: sum(1 for kw in debate_keywords if kw in query_text) / len(debate_keywords),
            QueryMode.QUOTES: sum(1 for kw in quotes_keywords if kw in query_text) / len(quotes_keywords)
        }

        return mode_scores

    def _determine_validation_recommendation(self, mode_scores: Dict[QueryMode, float],
                                             confidence: float) -> tuple[bool, Optional[ValidationType]]:
        """Determine if validation is recommended and what type."""
        validation_recommended = False
        suggested_validation_type = None

        if confidence < 0.6:
            validation_recommended = True
            suggested_validation_type = ValidationType.COMPREHENSIVE
        elif any(score > 0.3 for score in [
            mode_scores[QueryMode.FEATURES],
            mode_scores[QueryMode.TRADEOFFS],
            mode_scores[QueryMode.SCENARIOS]
        ]):
            validation_recommended = True
            suggested_validation_type = ValidationType.USER_GUIDED
        elif mode_scores[QueryMode.FACTS] > 0.6:
            suggested_validation_type = ValidationType.BASIC

        return validation_recommended, suggested_validation_type

    def _generate_compatibility_recommendations(self, confidence: float, mode_scores: Dict[QueryMode, float],
                                                validation_recommended: bool,
                                                suggested_validation_type: Optional[ValidationType]) -> List[str]:
        """Generate recommendations based on compatibility analysis."""
        recommendations = []

        if confidence < 0.4:
            recommendations.append("建议使用车辆规格查询模式（默认模式）")

        if validation_recommended:
            recommendations.append(f"建议启用{suggested_validation_type.value}验证以提高答案质量")

        if mode_scores[QueryMode.FACTS] > 0.3:
            recommendations.append("查询包含具体规格问题，车辆规格查询模式最适合")

        return recommendations