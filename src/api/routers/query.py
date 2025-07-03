import uuid
import json
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException
import logging
import numpy as np

from src.core.background.job_chain import job_chain, JobType
from src.core.orchestration.job_tracker import job_tracker
from src.models.schema import (
    # REMOVED: QueryRequest, QueryResponse, BackgroundJobResponse (legacy)
    # UNIFIED: Only enhanced query models
    EnhancedQueryRequest,
    EnhancedQueryResponse,
    EnhancedBackgroundJobResponse,
    QueryMode,
    QueryModeConfig,
    SystemCapabilities,
    QueryValidationResult
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Query mode configurations (unchanged)
QUERY_MODE_CONFIGS = {
    QueryMode.FACTS: QueryModeConfig(
        mode=QueryMode.FACTS,
        icon="📌",
        name="车辆规格查询",
        description="验证具体的车辆规格参数",
        use_case="查询确切的技术规格、配置信息",
        two_layer=False,  # UPDATED: Facts mode is now direct
        examples=[
            "2023年宝马X5的后备箱容积是多少？",
            "特斯拉Model 3的充电速度参数",
            "奔驰E级使用什么轮胎型号？"
        ]
    ),
    QueryMode.FEATURES: QueryModeConfig(
        mode=QueryMode.FEATURES,
        icon="💡",
        name="新功能建议",
        description="评估是否应该添加某项功能",
        use_case="产品决策，功能规划",
        two_layer=True,
        examples=[
            "是否应该为电动车增加氛围灯功能？",
            "增加模拟引擎声音对用户体验的影响",
            "AR抬头显示器值得投资吗？"
        ]
    ),
    QueryMode.TRADEOFFS: QueryModeConfig(
        mode=QueryMode.TRADEOFFS,
        icon="🧾",
        name="权衡利弊分析",
        description="分析设计选择的优缺点",
        use_case="设计决策，技术选型",
        two_layer=True,
        examples=[
            "使用模拟声音 vs 自然静音的利弊",
            "移除物理按键的优缺点分析",
            "大屏幕 vs 传统仪表盘的对比"
        ]
    ),
    QueryMode.SCENARIOS: QueryModeConfig(
        mode=QueryMode.SCENARIOS,
        icon="🧩",
        name="用户场景分析",
        description="评估功能在实际使用场景中的表现",
        use_case="用户体验设计，产品规划",
        two_layer=True,
        examples=[
            "长途旅行时这个功能如何表现？",
            "家庭用户在日常通勤中的体验如何？",
            "寒冷气候下的性能表现分析"
        ]
    ),
    QueryMode.DEBATE: QueryModeConfig(
        mode=QueryMode.DEBATE,
        icon="🗣️",
        name="多角色讨论",
        description="模拟不同角色的观点和讨论",
        use_case="决策支持，全面评估",
        two_layer=False,
        examples=[
            "产品经理、工程师和用户代表如何看待自动驾驶功能？",
            "不同团队对电池技术路线的观点",
            "关于车内空间设计的多方讨论"
        ]
    ),
    QueryMode.QUOTES: QueryModeConfig(
        mode=QueryMode.QUOTES,
        icon="🔍",
        name="原始用户评论",
        description="提取相关的用户评论和反馈",
        use_case="市场研究，用户洞察",
        two_layer=False,
        examples=[
            "用户对续航里程的真实评价",
            "关于内饰质量的用户反馈",
            "充电体验的用户评论摘录"
        ]
    )
}


async def handle_unified_query_logic(request: EnhancedQueryRequest) -> EnhancedBackgroundJobResponse:
    """
    UNIFIED: Single query handling function for all query modes.
    Facts mode serves as the default, replacing old normal queries.
    """
    try:
        job_id = str(uuid.uuid4())

        # Validate query mode
        if request.query_mode not in QUERY_MODE_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported query mode: {request.query_mode}"
            )

        mode_config = QUERY_MODE_CONFIGS[request.query_mode]

        # Determine complexity level
        complexity_map = {
            QueryMode.FACTS: "simple",      # UNIFIED: Facts is the new "normal"
            QueryMode.FEATURES: "moderate",
            QueryMode.TRADEOFFS: "complex",
            QueryMode.SCENARIOS: "complex",
            QueryMode.DEBATE: "complex",
            QueryMode.QUOTES: "simple"
        }
        complexity = complexity_map.get(request.query_mode, "simple")

        # Estimate processing time based on mode
        time_estimates = {
            QueryMode.FACTS: 10,        # OPTIMIZED: Faster for facts (was 15)
            QueryMode.FEATURES: 30,
            QueryMode.TRADEOFFS: 45,
            QueryMode.SCENARIOS: 40,
            QueryMode.DEBATE: 50,
            QueryMode.QUOTES: 20
        }
        estimated_time = time_estimates.get(request.query_mode, 20)

        # Create unified job record
        job_tracker.create_job(
            job_id=job_id,
            job_type="llm_inference",
            metadata={
                "query": request.query,
                "metadata_filter": request.metadata_filter,
                "top_k": request.top_k,
                "query_mode": request.query_mode.value,
                "has_custom_template": request.prompt_template is not None,
                "complexity_level": complexity,
                "estimated_time": estimated_time,
                "mode_name": mode_config.name,
                "unified_system": True  # Flag to indicate unified system
            }
        )

        # Start job chain with enhanced data
        job_chain.start_job_chain(
            job_id=job_id,
            job_type=JobType.LLM_INFERENCE,
            initial_data={
                "query": request.query,
                "metadata_filter": request.metadata_filter,
                "query_mode": request.query_mode.value,
                "custom_template": request.prompt_template
            }
        )

        return EnhancedBackgroundJobResponse(
            message=f"Query processing in '{mode_config.name}' mode",
            job_id=job_id,
            job_type="llm_inference",
            query_mode=request.query_mode,
            expected_processing_time=estimated_time,
            status="processing",
            complexity_level=complexity
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting unified query processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


# UNIFIED ENDPOINTS - Remove old normal query endpoints

@router.post("/", response_model=EnhancedBackgroundJobResponse)
async def unified_query_with_slash(request: EnhancedQueryRequest) -> EnhancedBackgroundJobResponse:
    """UNIFIED: Single query endpoint with trailing slash"""
    return await handle_unified_query_logic(request)


@router.post("", response_model=EnhancedBackgroundJobResponse)
async def unified_query_without_slash(request: EnhancedQueryRequest) -> EnhancedBackgroundJobResponse:
    """UNIFIED: Single query endpoint without trailing slash"""
    return await handle_unified_query_logic(request)


# REMOVED: Old enhanced query endpoint - now this IS the main endpoint
# @router.post("/enhanced", ...)  # No longer needed


@router.get("/modes", response_model=List[QueryModeConfig])
async def get_query_modes() -> List[QueryModeConfig]:
    """Get available query modes and their configurations"""
    return list(QUERY_MODE_CONFIGS.values())


@router.get("/modes/{mode}", response_model=QueryModeConfig)
async def get_query_mode(mode: QueryMode) -> QueryModeConfig:
    """Get configuration for a specific query mode"""
    if mode not in QUERY_MODE_CONFIGS:
        raise HTTPException(
            status_code=404,
            detail=f"Query mode '{mode}' not found"
        )
    return QUERY_MODE_CONFIGS[mode]


@router.get("/modes/default", response_model=QueryModeConfig)
async def get_default_query_mode() -> QueryModeConfig:
    """Get the default query mode (Facts - replaces normal queries)"""
    return QUERY_MODE_CONFIGS[QueryMode.FACTS]


@router.post("/validate", response_model=QueryValidationResult)
async def validate_query(request: EnhancedQueryRequest) -> QueryValidationResult:
    """Validate a query for mode compatibility"""
    try:
        query_text = request.query.lower()

        # Mode compatibility scoring
        compatibility_scores = {}

        # Facts mode: looks for specific questions (DEFAULT/FALLBACK)
        facts_keywords = ["什么", "多少", "哪个", "规格", "参数", "尺寸", "重量"]
        facts_score = sum(1 for kw in facts_keywords if kw in query_text) / len(facts_keywords)
        compatibility_scores[QueryMode.FACTS] = facts_score > 0.2  # Lower threshold (more inclusive)

        # Features mode: looks for evaluation language
        features_keywords = ["应该", "值得", "建议", "增加", "功能", "是否"]
        features_score = sum(1 for kw in features_keywords if kw in query_text) / len(features_keywords)
        compatibility_scores[QueryMode.FEATURES] = features_score > 0.2

        # Tradeoffs mode: looks for comparison language
        tradeoffs_keywords = ["vs", "对比", "优缺点", "利弊", "比较", "选择"]
        tradeoffs_score = sum(1 for kw in tradeoffs_keywords if kw in query_text) / len(tradeoffs_keywords)
        compatibility_scores[QueryMode.TRADEOFFS] = tradeoffs_score > 0.2

        # Scenarios mode: looks for scenario language
        scenarios_keywords = ["场景", "情况下", "使用时", "体验", "用户", "日常"]
        scenarios_score = sum(1 for kw in scenarios_keywords if kw in query_text) / len(scenarios_keywords)
        compatibility_scores[QueryMode.SCENARIOS] = scenarios_score > 0.2

        # Debate mode: looks for perspective language
        debate_keywords = ["观点", "看法", "认为", "角度", "团队", "讨论"]
        debate_score = sum(1 for kw in debate_keywords if kw in query_text) / len(debate_keywords)
        compatibility_scores[QueryMode.DEBATE] = debate_score > 0.2

        # Quotes mode: looks for feedback language
        quotes_keywords = ["评价", "反馈", "评论", "用户说", "真实", "体验"]
        quotes_score = sum(1 for kw in quotes_keywords if kw in query_text) / len(quotes_keywords)
        compatibility_scores[QueryMode.QUOTES] = quotes_score > 0.2

        # Calculate overall scores
        mode_scores = {
            QueryMode.FACTS: max(facts_score, 0.4),  # BOOST: Facts always has decent score
            QueryMode.FEATURES: features_score,
            QueryMode.TRADEOFFS: tradeoffs_score,
            QueryMode.SCENARIOS: scenarios_score,
            QueryMode.DEBATE: debate_score,
            QueryMode.QUOTES: quotes_score
        }

        # Suggest best mode (Facts is good fallback)
        suggested_mode = max(mode_scores.items(), key=lambda x: x[1])[0]
        confidence = max(mode_scores.values())

        # Generate recommendations
        recommendations = []
        if confidence < 0.4:
            recommendations.append("建议使用车辆规格查询模式（默认模式）")

        if facts_score > 0.3:
            recommendations.append("查询包含具体规格问题，车辆规格查询模式最适合")

        if any(score > 0.3 for score in [features_score, tradeoffs_score, scenarios_score, debate_score]):
            recommendations.append("查询适合深度分析，可考虑使用智能分析模式")

        return QueryValidationResult(
            is_valid=True,
            mode_compatibility=compatibility_scores,
            recommendations=recommendations,
            suggested_mode=suggested_mode,
            confidence_score=confidence
        )

    except Exception as e:
        logger.error(f"Error validating query: {str(e)}")
        return QueryValidationResult(
            is_valid=False,
            mode_compatibility={},
            recommendations=["查询验证失败，将使用默认的车辆规格查询模式"],
            suggested_mode=QueryMode.FACTS,  # DEFAULT FALLBACK
            confidence_score=0.0
        )


@router.get("/results/{job_id}", response_model=Optional[EnhancedQueryResponse])
async def get_query_result(job_id: str) -> Optional[EnhancedQueryResponse]:
    """UNIFIED: Get query results (always enhanced format)"""
    job_data = job_tracker.get_job(job_id)

    if not job_data:
        raise HTTPException(
            status_code=404,
            detail=f"Job with ID {job_id} not found"
        )

    status = job_data.get("status", "")
    metadata = job_data.get("metadata", {})
    result = job_data.get("result", {})

    # Parse result if it's a JSON string
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except:
            result = {"answer": result}

    # Get query mode (default to FACTS if missing)
    query_mode = metadata.get("query_mode", "facts")
    try:
        query_mode_enum = QueryMode(query_mode)
    except ValueError:
        query_mode_enum = QueryMode.FACTS  # Fallback to facts

    if status == "completed":
        # Get documents and clean them
        documents = result.get("documents", [])
        cleaned_documents = []

        # NEW: Calculate processing statistics
        processing_stats = calculate_processing_statistics(documents)

        for doc_data in documents:
            if isinstance(doc_data, dict):
                doc_metadata = doc_data.get("metadata", {})
                cleaned_metadata = {}

                for key, value in doc_metadata.items():
                    if isinstance(value, (np.floating, np.float32, np.float64)):
                        cleaned_metadata[key] = float(value)
                    elif isinstance(value, (np.integer, np.int32, np.int64)):
                        cleaned_metadata[key] = int(value)
                    elif isinstance(value, np.ndarray):
                        cleaned_metadata[key] = value.tolist()
                    else:
                        cleaned_metadata[key] = value

                cleaned_doc = {
                    "id": doc_data.get("id", ""),
                    "content": doc_data.get("content", ""),  # CRITICAL: Must include embedded metadata
                    "metadata": cleaned_metadata,  # CRITICAL: Must include all processing metadata
                    "relevance_score": float(doc_data.get("relevance_score", 0.0))
                }
                cleaned_documents.append(cleaned_doc)

        # Clean the answer
        answer = result.get("answer", "")
        if answer.startswith("</think>\n\n"):
            answer = answer.replace("</think>\n\n", "").strip()
        if answer.startswith("<think>") and "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()

        # Parse structured sections for two-layer modes
        analysis_structure = None
        mode_metadata = {
            "processing_mode": query_mode,
            "complexity_level": metadata.get("complexity_level", "simple"),
            "mode_name": metadata.get("mode_name", ""),
            "unified_system": True,
            "processing_stats": processing_stats  # NEW: Add processing statistics
        }

        mode_config = QUERY_MODE_CONFIGS.get(query_mode_enum)
        if mode_config and mode_config.two_layer:
            # Parse structured sections
            analysis_structure = parse_structured_answer(answer, query_mode)
            mode_metadata["structured_sections"] = list(analysis_structure.keys()) if analysis_structure else []

        return EnhancedQueryResponse(
            query=result.get("query", metadata.get("query", "")),
            answer=answer,
            documents=cleaned_documents,
            query_mode=query_mode_enum,
            analysis_structure=analysis_structure,
            metadata_filters_used=metadata.get("metadata_filter"),
            execution_time=result.get("execution_time", 0),
            status=status,
            job_id=job_id,
            mode_metadata=mode_metadata
        )

    elif status == "failed":
        # Create error response
        error_msg = job_data.get('error', 'Unknown error')

        return EnhancedQueryResponse(
            query=metadata.get("query", ""),
            answer=f"Error: {error_msg}",
            documents=[],
            query_mode=query_mode_enum,
            metadata_filters_used=metadata.get("metadata_filter"),
            execution_time=0,
            status=status,
            job_id=job_id
        )
    else:
        # Still processing
        chain_status = job_chain.get_job_chain_status(job_id)
        progress_msg = "Processing query..."

        if chain_status:
            current_task = chain_status.get("current_task", "unknown")
            mode_name = metadata.get("mode_name", query_mode)
            progress_msg = f"Processing {mode_name}... (Current: {current_task})"

        return EnhancedQueryResponse(
            query=metadata.get("query", ""),
            answer=progress_msg,
            documents=[],
            query_mode=query_mode_enum,
            metadata_filters_used=metadata.get("metadata_filter"),
            execution_time=0,
            status=status,
            job_id=job_id
        )

def calculate_processing_statistics(documents: List[Dict]) -> Dict[str, Any]:
    """Calculate processing statistics for UI display with ENHANCED PROCESSOR support."""
    if not documents:
        return {}

    total_docs = len(documents)
    docs_with_embedded = 0
    docs_with_vehicle = 0
    unique_vehicles = set()
    unique_sources = set()
    total_metadata_fields = 0

    for doc in documents:
        metadata = doc.get("metadata", {})
        content = doc.get("content", "")

        # ✅ CHECK FOR EMBEDDED METADATA (ENHANCED PROCESSOR PATTERNS)
        import re
        embedded_metadata_pattern = r'【[^】]+】'
        embedded_matches = re.findall(embedded_metadata_pattern, content)

        if embedded_matches:
            docs_with_embedded += 1
            total_metadata_fields += len(embedded_matches)
            logger.info(f"✅ Found {len(embedded_matches)} embedded metadata patterns in document")
        else:
            # ✅ ALSO CHECK METADATA FLAGS
            if metadata.get('metadataInjected') or metadata.get('metadata_injected'):
                docs_with_embedded += 1
                logger.info(f"✅ Document marked as having metadata injection")

        # ✅ CHECK VEHICLE INFO (ENHANCED PROCESSOR FIELDS)
        if (metadata.get('vehicleDetected') or
                metadata.get('has_vehicle_info') or
                metadata.get('vehicleModel') or
                metadata.get('model')):
            docs_with_vehicle += 1

            # Track unique vehicles
            manufacturer = metadata.get('manufacturer', '')
            model = metadata.get('vehicleModel') or metadata.get('model', '')

            if model:
                if manufacturer:
                    vehicle_name = f"{manufacturer} {model}"
                else:
                    vehicle_name = model
                unique_vehicles.add(vehicle_name)

        # Track sources (enhanced processor uses 'source' field)
        source = metadata.get('source') or metadata.get('sourcePlatform')
        if source:
            unique_sources.add(source)

    return {
        "total_documents": total_docs,
        "documents_with_metadata": docs_with_embedded,
        "documents_with_vehicle_info": docs_with_vehicle,
        "unique_vehicles_detected": list(unique_vehicles),
        "unique_sources": list(unique_sources),
        "metadata_injection_rate": docs_with_embedded / total_docs if total_docs > 0 else 0,
        "vehicle_detection_rate": docs_with_vehicle / total_docs if total_docs > 0 else 0,
        "avg_metadata_fields_per_doc": total_metadata_fields / max(docs_with_embedded, 1),
        # ✅ ADD ENHANCED PROCESSING INDICATORS
        "enhanced_processing_detected": any(
            doc.get("metadata", {}).get("enhanced_processing_used") or
            doc.get("metadata", {}).get("processing_method") == "enhanced_transcript_processor"
            for doc in documents
        )
    }

def parse_structured_answer(answer: str, query_mode: str) -> Optional[Dict[str, str]]:
    """Parse structured answer into sections based on query mode"""
    section_patterns = {
        "features": ["【实证分析】", "【策略推理】"],
        "tradeoffs": ["【文档支撑】", "【利弊分析】"],
        "scenarios": ["【文档场景】", "【场景推理】"]
    }

    if query_mode not in section_patterns:
        return None

    headers = section_patterns[query_mode]
    sections = {}
    current_section = None
    current_content = []

    lines = answer.split('\n')

    for line in lines:
        line = line.strip()

        # Check if this line is a section header
        found_header = None
        for header in headers:
            if header in line:
                found_header = header
                break

        if found_header:
            # Save previous section if exists
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()

            # Start new section
            current_section = found_header
            current_content = []
        elif current_section and line:
            current_content.append(line)

    # Save the last section
    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content).strip()

    return sections if sections else None


@router.get("/capabilities", response_model=SystemCapabilities)
async def get_system_capabilities() -> SystemCapabilities:
    """Get system capabilities and current status"""
    try:
        # Get current system load
        current_load = {}
        try:
            queue_status = job_chain.get_queue_status()
            for queue_name, status in queue_status.items():
                if isinstance(status, dict):
                    waiting = status.get("waiting_tasks", 0)
                    current_load[queue_name] = waiting
        except:
            current_load = {"inference_tasks": 0, "embedding_tasks": 0}

        # Calculate estimated response times based on load
        base_times = {
            QueryMode.FACTS: 10,        # OPTIMIZED: Faster facts mode
            QueryMode.FEATURES: 30,
            QueryMode.TRADEOFFS: 45,
            QueryMode.SCENARIOS: 40,
            QueryMode.DEBATE: 50,
            QueryMode.QUOTES: 20
        }

        # Add delay based on queue load
        queue_delay = current_load.get("inference_tasks", 0) * 8  # Reduced delay
        estimated_times = {mode: base_time + queue_delay for mode, base_time in base_times.items()}

        # System status check
        try:
            from src.ui.api_client import api_request
            health_response = api_request("/health", silent=True)
            if health_response and health_response.get("status") == "healthy":
                system_status = "healthy"
            else:
                system_status = "degraded"
        except:
            system_status = "degraded"

        return SystemCapabilities(
            supported_modes=list(QUERY_MODE_CONFIGS.values()),
            current_load=current_load,
            estimated_response_times=estimated_times,
            feature_flags={
                "unified_query_system": True,      # NEW FLAG
                "enhanced_queries": True,
                "structured_analysis": True,
                "multi_perspective": True,
                "user_quotes": True,
                "query_validation": True,
                "facts_as_default": True           # NEW FLAG
            },
            system_status=system_status
        )

    except Exception as e:
        logger.error(f"Error getting system capabilities: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting system capabilities: {str(e)}"
        )


# LEGACY SUPPORT ENDPOINTS (can be removed later)

@router.get("/manufacturers", response_model=List[str])
async def get_manufacturers() -> List[str]:
    """Get a list of available manufacturers."""
    return [
        "Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes",
        "Audi", "Volkswagen", "Nissan", "Hyundai", "Kia", "Subaru"
    ]


@router.get("/models", response_model=List[str])
async def get_models(manufacturer: Optional[str] = None) -> List[str]:
    """Get a list of available models, optionally filtered by manufacturer."""
    if manufacturer == "Toyota":
        return ["Camry", "Corolla", "RAV4", "Highlander", "Tacoma"]
    elif manufacturer == "Honda":
        return ["Civic", "Accord", "CR-V", "Pilot", "Odyssey"]
    elif manufacturer == "Ford":
        return ["Mustang", "F-150", "Escape", "Explorer", "Edge"]
    elif manufacturer == "BMW":
        return ["3 Series", "5 Series", "X3", "X5", "i4"]
    else:
        return ["Model S", "Model 3", "Model X", "Model Y", "Cybertruck"]


@router.get("/queue-status", response_model=Dict[str, Any])
async def get_queue_status() -> Dict[str, Any]:
    """Get status of the job chain queue system."""
    try:
        return job_chain.get_queue_status()
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting queue status: {str(e)}"
        )

@router.post("/debug-retrieval", response_model=Dict[str, Any])
async def debug_document_retrieval(request: Dict[str, str]) -> Dict[str, Any]:
    """
    Debug endpoint to retrieve documents from vector store for browsing.
    Used by the document browser page.
    """
    try:
        from src.core.background.models import get_vector_store

        query = request.get("query", "document")
        limit = min(int(request.get("limit", 100)), 500)  # Cap at 500

        vector_store = get_vector_store()

        # Get documents using similarity search with a generic query
        results = vector_store.similarity_search_with_score(
            query=query,
            k=limit,
            metadata_filter=None
        )

        # Format results for UI consumption with enhanced metadata
        documents = []
        for doc, score in results:
            # Clean metadata to ensure JSON serialization
            cleaned_metadata = {}
            for key, value in doc.metadata.items():
                if isinstance(value, (np.floating, np.float32, np.float64)):
                    cleaned_metadata[key] = float(value)
                elif isinstance(value, (np.integer, np.int32, np.int64)):
                    cleaned_metadata[key] = int(value)
                elif isinstance(value, np.ndarray):
                    cleaned_metadata[key] = value.tolist()
                else:
                    cleaned_metadata[key] = value

            documents.append({
                "content": doc.page_content,  # CRITICAL: Must include full content with embedded metadata
                "metadata": cleaned_metadata,  # CRITICAL: Must include all metadata fields
                "relevance_score": float(score) if isinstance(score, (np.floating, np.float32, np.float64)) else score
            })

        # NEW: Calculate metadata statistics for the document browser
        processing_stats = calculate_processing_statistics(documents)

        logger.info(f"Debug retrieval returned {len(documents)} documents")

        return {
            "documents": documents,
            "total_found": len(documents),
            "query_used": query,
            "processing_stats": processing_stats  # NEW: Add processing statistics
        }

    except Exception as e:
        logger.error(f"Debug retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Document retrieval failed: {str(e)}"
        )
