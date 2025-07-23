import uuid
import json
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Path, Query as FastAPIQuery
import logging
import numpy as np
from datetime import datetime

from src.core.orchestration.job_chain import job_chain, JobType
from src.core.orchestration.job_tracker import job_tracker
from src.models.schema import (
    # Enhanced query models with validation
    EnhancedQueryRequest,
    EnhancedQueryResponse,
    EnhancedBackgroundJobResponse,

    # Core query models
    QueryMode,
    QueryModeConfig,
    SystemCapabilities,
    QueryValidationResult,
    DocumentResponse
)

# Import validation models and components
from src.core.validation.models.validation_models import (
    ValidationWorkflow,
    ValidationStatus,
    ValidationStep,
    ValidationType,
    ValidationStepResult,
    ValidationStepType,
    UserChoice,
    LearningCredit,
    ValidationUpdate
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Query mode configurations (unchanged from existing)
QUERY_MODE_CONFIGS = {
    QueryMode.FACTS: QueryModeConfig(
        mode=QueryMode.FACTS,
        icon="📌",
        name="车辆规格查询",
        description="验证具体的车辆规格参数",
        use_case="查询确切的技术规格、配置信息",
        two_layer=False,
        is_default=True,
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

# In-memory validation store (in production, use proper database)
validation_workflows: Dict[str, ValidationWorkflow] = {}


def create_validation_workflow(job_id: str, validation_config: Dict[str, Any]) -> ValidationWorkflow:
    """Create a new validation workflow data model."""

    validation_id = str(uuid.uuid4())
    validation_type = ValidationType(validation_config.get("validation_type", "basic"))

    workflow = ValidationWorkflow(
        validation_id=validation_id,
        job_id=job_id,
        validation_type=validation_type,
        overall_status=ValidationStatus.PENDING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        steps=[],
        current_step=None,
        user_choices=[],
        awaiting_user_input=False,
        user_input_prompt=None,
        overall_confidence=None,
        total_duration_seconds=None,
        validation_passed=None,
        issues_identified=[],
        final_recommendations=[],
        metadata={
            "confidence_threshold": validation_config.get("confidence_threshold", 0.7),
            "require_user_approval": validation_config.get("require_user_approval", False),
            "auto_approve_high_confidence": validation_config.get("auto_approve_high_confidence", True),
            "high_confidence_threshold": validation_config.get("high_confidence_threshold", 0.9)
        }
    )

    # Initialize validation steps based on type
    if validation_type == ValidationType.BASIC:
        steps = [
            ValidationStep.CONFIDENCE_ANALYSIS,
            ValidationStep.ANSWER_GENERATION
        ]
    elif validation_type == ValidationType.COMPREHENSIVE:
        steps = [
            ValidationStep.DOCUMENT_RETRIEVAL,
            ValidationStep.RELEVANCE_SCORING,
            ValidationStep.CONFIDENCE_ANALYSIS,
            ValidationStep.USER_VERIFICATION,
            ValidationStep.ANSWER_GENERATION,
            ValidationStep.FINAL_REVIEW
        ]
    elif validation_type == ValidationType.USER_GUIDED:
        steps = [
            ValidationStep.DOCUMENT_RETRIEVAL,
            ValidationStep.RELEVANCE_SCORING,
            ValidationStep.USER_VERIFICATION,
            ValidationStep.ANSWER_GENERATION
        ]
    else:
        steps = [ValidationStep.CONFIDENCE_ANALYSIS]

    # Create ValidationStepResult objects for each step
    workflow.steps = [
        ValidationStepResult(
            step=step,
            status=ValidationStatus.PENDING,
            started_at=None,
            completed_at=None,
            duration_seconds=None,
            confidence_score=None,
            issues_found=[],
            recommendations=[],
            metadata={}
        )
        for step in steps
    ]

    if workflow.steps:
        workflow.current_step = workflow.steps[0].step

    # Store in validation workflows
    validation_workflows[validation_id] = workflow

    logger.info(f"Created validation workflow {validation_id} for job {job_id} with type {validation_type}")

    return workflow


async def handle_unified_query_with_validation(request: EnhancedQueryRequest) -> EnhancedBackgroundJobResponse:
    """Handle query processing with integrated validation workflow."""

    try:
        job_id = str(uuid.uuid4())

        # Validate query mode
        if request.query_mode not in QUERY_MODE_CONFIGS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported query mode: {request.query_mode}"
            )

        mode_config = QUERY_MODE_CONFIGS[request.query_mode]

        # Check if validation is enabled
        validation_config = getattr(request, 'validation_config', None)
        validation_enabled = validation_config and validation_config.enabled if validation_config else False
        validation_workflow = None
        validation_id = None

        if validation_enabled:
            # Create validation workflow as a data model
            validation_workflow = create_validation_workflow(job_id, validation_config.dict())
            validation_id = validation_workflow.validation_id

            logger.info(f"Validation enabled for job {job_id} with workflow {validation_id}")

        # Determine complexity and processing time
        complexity_map = {
            QueryMode.FACTS: "simple",
            QueryMode.FEATURES: "moderate",
            QueryMode.TRADEOFFS: "complex",
            QueryMode.SCENARIOS: "complex",
            QueryMode.DEBATE: "complex",
            QueryMode.QUOTES: "simple"
        }
        complexity = complexity_map.get(request.query_mode, "simple")

        # Estimate processing time (add time for validation)
        base_times = {
            QueryMode.FACTS: 10,
            QueryMode.FEATURES: 30,
            QueryMode.TRADEOFFS: 45,
            QueryMode.SCENARIOS: 40,
            QueryMode.DEBATE: 50,
            QueryMode.QUOTES: 20
        }

        estimated_time = base_times.get(request.query_mode, 20)
        if validation_enabled:
            validation_overhead = {
                ValidationType.BASIC: 5,
                ValidationType.COMPREHENSIVE: 15,
                ValidationType.USER_GUIDED: 10
            }
            estimated_time += validation_overhead.get(validation_workflow.validation_type, 5)

        # Create enhanced job metadata with validation info
        job_metadata = {
            "query": request.query,
            "metadata_filter": request.metadata_filter,
            "top_k": request.top_k,
            "query_mode": request.query_mode.value,
            "has_custom_template": request.prompt_template is not None,
            "complexity_level": complexity,
            "estimated_time": estimated_time,
            "mode_name": mode_config.name,
            "validation_enabled": validation_enabled,
            "validation_id": validation_id,
            "validation_type": validation_workflow.validation_type.value if validation_workflow else None,
            "unified_system": True
        }

        # Create job record
        job_tracker.create_job(
            job_id=job_id,
            job_type="llm_inference",
            metadata=job_metadata
        )

        # Start job chain with validation integration
        initial_data = {
            "query": request.query,
            "metadata_filter": request.metadata_filter,
            "query_mode": request.query_mode.value,
            "custom_template": request.prompt_template,
            "validation_enabled": validation_enabled,
            "validation_id": validation_id,
            "validation_config": validation_config.dict() if validation_config else None
        }

        # Determine job type based on validation
        if validation_enabled and validation_workflow.validation_type == ValidationType.COMPREHENSIVE:
            job_type = JobType.COMPREHENSIVE_VALIDATION
        else:
            job_type = JobType.LLM_INFERENCE

        job_chain.start_job_chain(
            job_id=job_id,
            job_type=job_type,
            initial_data=initial_data
        )

        return EnhancedBackgroundJobResponse(
            message=f"Query processing in '{mode_config.name}' mode" +
                    (f" with {validation_workflow.validation_type.value} validation" if validation_enabled else ""),
            job_id=job_id,
            job_type=job_type.value,
            query_mode=request.query_mode,
            expected_processing_time=estimated_time,
            status="processing",
            complexity_level=complexity,
            validation_enabled=validation_enabled,
            validation_id=validation_id,
            validation_type=validation_workflow.validation_type if validation_workflow else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting query processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


# ==================== MAIN QUERY ENDPOINTS ====================

@router.post("/", response_model=EnhancedBackgroundJobResponse)
async def submit_query_with_slash(request: EnhancedQueryRequest) -> EnhancedBackgroundJobResponse:
    """Submit a query for processing with optional validation."""
    return await handle_unified_query_with_validation(request)


@router.post("", response_model=EnhancedBackgroundJobResponse)
async def submit_query_without_slash(request: EnhancedQueryRequest) -> EnhancedBackgroundJobResponse:
    """Submit a query for processing with optional validation."""
    return await handle_unified_query_with_validation(request)


@router.get("/{job_id}", response_model=Optional[EnhancedQueryResponse])
async def get_query_result(job_id: str = Path(..., description="Job ID")) -> Optional[EnhancedQueryResponse]:
    """Get query results with integrated validation data."""

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

    # Get query mode
    query_mode = metadata.get("query_mode", "facts")
    try:
        query_mode_enum = QueryMode(query_mode)
    except ValueError:
        query_mode_enum = QueryMode.FACTS

    # Get validation workflow if it exists
    validation_id = metadata.get("validation_id")
    validation_workflow = None
    validation_summary = None

    if validation_id and validation_id in validation_workflows:
        validation_workflow = validation_workflows[validation_id]
        validation_summary = {
            "validation_id": validation_id,
            "status": validation_workflow.overall_status.value,
            "type": validation_workflow.validation_type.value,
            "confidence": validation_workflow.overall_confidence,
            "passed": validation_workflow.validation_passed,
            "steps_completed": len([s for s in validation_workflow.steps if s.status == ValidationStatus.COMPLETED]),
            "total_steps": len(validation_workflow.steps),
            "awaiting_user_input": validation_workflow.awaiting_user_input,
            "current_step": validation_workflow.current_step.value if validation_workflow.current_step else None,
            "issues_identified": len(validation_workflow.issues_identified),
            "recommendations": len(validation_workflow.final_recommendations)
        }

    if status == "completed":
        # Process documents and calculate stats
        documents = result.get("documents", [])
        cleaned_documents = []
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

                cleaned_doc = DocumentResponse(
                    id=doc_data.get("id", ""),
                    content=doc_data.get("content", ""),
                    metadata=cleaned_metadata,
                    relevance_score=float(doc_data.get("relevance_score", 0.0))
                )
                cleaned_documents.append(cleaned_doc)

        # Clean answer
        answer = result.get("answer", "")
        if answer.startswith("</think>\n\n"):
            answer = answer.replace("</think>\n\n", "").strip()
        if answer.startswith("<think>") and "</think>" in answer:
            answer = answer.split("</think>")[-1].strip()

        # Parse structured sections for two-layer modes
        analysis_structure = None
        mode_config = QUERY_MODE_CONFIGS.get(query_mode_enum)
        if mode_config and mode_config.two_layer:
            analysis_structure = parse_structured_answer(answer, query_mode)

        mode_metadata = {
            "processing_mode": query_mode,
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
            status=status,
            job_id=job_id,
            mode_metadata=mode_metadata,
            validation=validation_workflow,
            validation_summary=validation_summary
        )

    elif status == "failed":
        error_msg = job_data.get('error', 'Unknown error')
        return EnhancedQueryResponse(
            query=metadata.get("query", ""),
            answer=f"Error: {error_msg}",
            documents=[],
            query_mode=query_mode_enum,
            metadata_filters_used=metadata.get("metadata_filter"),
            execution_time=0,
            status=status,
            job_id=job_id,
            validation=validation_workflow,
            validation_summary=validation_summary
        )
    else:
        # Still processing
        chain_status = job_chain.get_job_chain_status(job_id)
        progress_msg = "Processing query..."

        if chain_status:
            current_task = chain_status.get("current_task", "unknown")
            mode_name = metadata.get("mode_name", query_mode)

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
            status=status,
            job_id=job_id,
            validation=validation_workflow,
            validation_summary=validation_summary
        )


# ==================== VALIDATION SUB-RESOURCE ENDPOINTS ====================

@router.get("/{job_id}/validation", response_model=Optional[Dict[str, Any]])
async def get_validation_progress(
        job_id: str = Path(..., description="Job ID")
) -> Optional[Dict[str, Any]]:
    """Get validation progress for a job."""

    job_data = job_tracker.get_job(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    validation_id = job_data.get("metadata", {}).get("validation_id")

    if not validation_id or validation_id not in validation_workflows:
        raise HTTPException(status_code=404, detail=f"No validation workflow found for job {job_id}")

    validation = validation_workflows[validation_id]

    # Calculate progress
    completed_steps = len([s for s in validation.steps if s.status == ValidationStatus.COMPLETED])
    total_steps = len(validation.steps)
    progress_percentage = (completed_steps / total_steps * 100) if total_steps > 0 else 0

    # Estimate completion time
    estimated_completion = None
    if validation.overall_status == ValidationStatus.IN_PROGRESS:
        remaining_steps = total_steps - completed_steps
        estimated_completion = remaining_steps * 10  # Rough estimate

    return {
        "validation_id": validation_id,
        "job_id": job_id,
        "validation_type": validation.validation_type.value,
        "overall_status": validation.overall_status.value,
        "current_step": validation.current_step.value if validation.current_step else None,
        "progress_percentage": progress_percentage,
        "steps_completed": completed_steps,
        "total_steps": total_steps,
        "overall_confidence": validation.overall_confidence,
        "awaiting_user_input": validation.awaiting_user_input,
        "user_input_prompt": validation.user_input_prompt,
        "estimated_completion_time": estimated_completion,
        "steps": [
            {
                "step": step.step.value,
                "status": step.status.value,
                "confidence_score": step.confidence_score,
                "duration_seconds": step.duration_seconds,
                "issues_found": len(step.issues_found),
                "recommendations": len(step.recommendations)
            }
            for step in validation.steps
        ],
        "issues_identified": validation.issues_identified,
        "final_recommendations": validation.final_recommendations
    }


@router.post("/{job_id}/validation/user-choice", response_model=Dict[str, Any])
async def submit_user_choice_for_validation(
        job_id: str = Path(..., description="Job ID"),
        choice_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Submit user choice for validation workflow."""

    job_data = job_tracker.get_job(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    validation_id = job_data.get("metadata", {}).get("validation_id")

    if not validation_id or validation_id not in validation_workflows:
        raise HTTPException(status_code=404, detail=f"No validation workflow found for job {job_id}")

    validation = validation_workflows[validation_id]

    if not validation.awaiting_user_input:
        raise HTTPException(status_code=400, detail="Validation is not awaiting user input")

    # Create user choice record
    user_choice = UserChoice(
        choice_id=str(uuid.uuid4()),
        step=validation.current_step,
        user_decision=choice_data.get("decision", "approve"),
        user_feedback=choice_data.get("feedback"),
        modified_data=choice_data.get("modified_data"),
        timestamp=datetime.now()
    )

    validation.user_choices.append(user_choice)
    validation.awaiting_user_input = False
    validation.user_input_prompt = None
    validation.updated_at = datetime.now()

    # Process the choice and update validation state
    choice_result = process_user_choice(validation, user_choice)

    # Update validation store
    validation_workflows[validation_id] = validation

    # Resume job chain with user choice
    user_choice_data = {
        "choice": choice_data.get("decision", "approve"),
        "validation_id": validation_id,
        "user_feedback": choice_data.get("feedback"),
        "modified_data": choice_data.get("modified_data")
    }

    try:
        job_chain.resume_from_user_input(job_id, user_choice_data)
    except Exception as e:
        logger.error(f"Failed to resume job chain: {str(e)}")
        # Continue anyway as we've updated the validation state

    return {
        "message": f"User choice '{choice_data.get('decision')}' processed successfully",
        "validation_id": validation_id,
        "choice_accepted": choice_result["accepted"],
        "next_step": choice_result.get("next_step"),
        "requires_user_input": validation.awaiting_user_input,
        "validation_status": validation.overall_status.value
    }


@router.post("/{job_id}/validation/restart", response_model=Dict[str, Any])
async def restart_validation_workflow(
        job_id: str = Path(..., description="Job ID"),
        restart_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Restart validation workflow from beginning or specific step."""

    job_data = job_tracker.get_job(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    validation_id = job_data.get("metadata", {}).get("validation_id")

    if not validation_id or validation_id not in validation_workflows:
        raise HTTPException(status_code=404, detail=f"No validation workflow found for job {job_id}")

    validation = validation_workflows[validation_id]

    # Reset validation state
    restart_from_step = None
    if restart_data and "step" in restart_data:
        try:
            restart_from_step = ValidationStep(restart_data["step"])
        except ValueError:
            restart_from_step = None

    if restart_from_step:
        # Reset from specific step
        restart_index = next(
            (i for i, s in enumerate(validation.steps) if s.step == restart_from_step),
            0
        )

        for i in range(restart_index, len(validation.steps)):
            step = validation.steps[i]
            step.status = ValidationStatus.PENDING
            step.started_at = None
            step.completed_at = None
            step.confidence_score = None
            step.issues_found = []
            step.recommendations = []

        validation.current_step = restart_from_step
    else:
        # Reset all steps
        for step in validation.steps:
            step.status = ValidationStatus.PENDING
            step.started_at = None
            step.completed_at = None
            step.confidence_score = None
            step.issues_found = []
            step.recommendations = []

        validation.current_step = validation.steps[0].step if validation.steps else None

    validation.overall_status = ValidationStatus.PENDING
    validation.awaiting_user_input = False
    validation.user_input_prompt = None
    validation.overall_confidence = None
    validation.validation_passed = None
    validation.issues_identified = []
    validation.final_recommendations = []
    validation.updated_at = datetime.now()

    # Update store
    validation_workflows[validation_id] = validation

    # Restart job chain
    restart_choice_data = {
        "choice": "restart_full",
        "validation_id": validation_id,
        "restart_from_step": restart_from_step.value if restart_from_step else None
    }

    try:
        job_chain.resume_from_user_input(job_id, restart_choice_data)
    except Exception as e:
        logger.error(f"Failed to restart job chain: {str(e)}")
        # Continue anyway as we've updated the validation state

    return {
        "message": f"Validation restarted from {restart_from_step.value if restart_from_step else 'beginning'}",
        "validation_id": validation_id,
        "restart_successful": True,
        "current_step": validation.current_step.value if validation.current_step else None,
        "validation_status": validation.overall_status.value
    }


@router.delete("/{job_id}/validation", response_model=Dict[str, str])
async def cancel_validation_workflow(
        job_id: str = Path(..., description="Job ID")
) -> Dict[str, str]:
    """Cancel validation workflow for a job."""

    job_data = job_tracker.get_job(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    validation_id = job_data.get("metadata", {}).get("validation_id")

    if not validation_id or validation_id not in validation_workflows:
        raise HTTPException(status_code=404, detail=f"No validation workflow found for job {job_id}")

    validation = validation_workflows[validation_id]
    validation.overall_status = ValidationStatus.CANCELLED
    validation.awaiting_user_input = False
    validation.updated_at = datetime.now()
    validation.completed_at = datetime.now()

    validation_workflows[validation_id] = validation

    # Note: Job will continue without validation

    return {"message": f"Validation workflow {validation_id} cancelled successfully"}


# ==================== UTILITY FUNCTIONS ====================

def process_user_choice(validation: ValidationWorkflow, user_choice: UserChoice) -> Dict[str, Any]:
    """Process user choice and update validation workflow state."""

    choice_result = {"accepted": True, "next_step": None}

    if user_choice.user_decision == "approve":
        # Move to next step
        current_step_index = next(
            (i for i, s in enumerate(validation.steps) if s.step == user_choice.step),
            -1
        )

        if current_step_index >= 0:
            # Mark current step as completed
            validation.steps[current_step_index].status = ValidationStatus.COMPLETED
            validation.steps[current_step_index].completed_at = datetime.now()

            # Move to next step if available
            if current_step_index < len(validation.steps) - 1:
                next_step = validation.steps[current_step_index + 1]
                validation.current_step = next_step.step
                choice_result["next_step"] = next_step.step.value
            else:
                # All steps completed
                validation.overall_status = ValidationStatus.COMPLETED
                validation.completed_at = datetime.now()
                validation.validation_passed = True

    elif user_choice.user_decision == "reject":
        # Mark step as failed
        current_step_index = next(
            (i for i, s in enumerate(validation.steps) if s.step == user_choice.step),
            -1
        )

        if current_step_index >= 0:
            validation.steps[current_step_index].status = ValidationStatus.FAILED
            validation.steps[current_step_index].completed_at = datetime.now()

        validation.overall_status = ValidationStatus.FAILED
        validation.validation_passed = False

        if user_choice.user_feedback:
            validation.issues_identified.append(f"User rejected {user_choice.step.value}: {user_choice.user_feedback}")

    elif user_choice.user_decision == "modify":
        # Handle modification
        if user_choice.modified_data:
            # Apply modifications and continue
            current_step_index = next(
                (i for i, s in enumerate(validation.steps) if s.step == user_choice.step),
                -1
            )

            if current_step_index >= 0:
                validation.steps[current_step_index].metadata.update(user_choice.modified_data)
                validation.steps[current_step_index].status = ValidationStatus.COMPLETED
                validation.steps[current_step_index].completed_at = datetime.now()

                # Move to next step
                if current_step_index < len(validation.steps) - 1:
                    next_step = validation.steps[current_step_index + 1]
                    validation.current_step = next_step.step
                    choice_result["next_step"] = next_step.step.value

    return choice_result


def calculate_processing_statistics(documents: List[Dict]) -> Dict[str, Any]:
    """Calculate processing statistics for UI display with enhanced processor support."""

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

        # Check for embedded metadata
        import re
        embedded_metadata_pattern = r'【[^】]+】'
        embedded_matches = re.findall(embedded_metadata_pattern, content)

        if embedded_matches:
            docs_with_embedded += 1
            total_metadata_fields += len(embedded_matches)
        elif metadata.get('metadataInjected') or metadata.get('metadata_injected'):
            docs_with_embedded += 1

        # Check vehicle info
        if (metadata.get('vehicleDetected') or
                metadata.get('has_vehicle_info') or
                metadata.get('vehicleModel') or
                metadata.get('model')):
            docs_with_vehicle += 1

            manufacturer = metadata.get('manufacturer', '')
            model = metadata.get('vehicleModel') or metadata.get('model', '')

            if model:
                vehicle_name = f"{manufacturer} {model}" if manufacturer else model
                unique_vehicles.add(vehicle_name)

        # Track sources
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
        "enhanced_processing_detected": any(
            doc.get("metadata", {}).get("enhanced_processing_used") or
            doc.get("metadata", {}).get("processing_method") == "enhanced_transcript_processor"
            for doc in documents
        )
    }


def parse_structured_answer(answer: str, query_mode: str) -> Optional[Dict[str, str]]:
    """Parse structured answer into sections based on query mode."""

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

        found_header = None
        for header in headers:
            if header in line:
                found_header = header
                break

        if found_header:
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()

            current_section = found_header
            current_content = []
        elif current_section and line:
            current_content.append(line)

    if current_section and current_content:
        sections[current_section] = '\n'.join(current_content).strip()

    return sections if sections else None


# ==================== EXISTING ENDPOINTS (unchanged) ====================

@router.get("/modes", response_model=List[QueryModeConfig])
async def get_query_modes() -> List[QueryModeConfig]:
    """Get available query modes and their configurations."""
    return list(QUERY_MODE_CONFIGS.values())


@router.get("/modes/{mode}", response_model=QueryModeConfig)
async def get_query_mode(mode: QueryMode) -> QueryModeConfig:
    """Get configuration for a specific query mode."""
    if mode not in QUERY_MODE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Query mode '{mode}' not found")
    return QUERY_MODE_CONFIGS[mode]


@router.get("/capabilities", response_model=SystemCapabilities)
async def get_system_capabilities() -> SystemCapabilities:
    """Get system capabilities and current status."""
    try:
        current_load = {}
        try:
            queue_status = job_chain.get_queue_status()
            for queue_name, status in queue_status.items():
                if isinstance(status, dict):
                    waiting = status.get("waiting_tasks", 0)
                    current_load[queue_name] = waiting
        except:
            current_load = {"inference_tasks": 0, "embedding_tasks": 0}

        base_times = {
            QueryMode.FACTS: 10,
            QueryMode.FEATURES: 30,
            QueryMode.TRADEOFFS: 45,
            QueryMode.SCENARIOS: 40,
            QueryMode.DEBATE: 50,
            QueryMode.QUOTES: 20
        }

        queue_delay = current_load.get("inference_tasks", 0) * 8
        estimated_times = {mode: base_time + queue_delay for mode, base_time in base_times.items()}

        try:
            from src.ui.api_client import api_request
            health_response = api_request("/health", silent=True)
            system_status = "healthy" if health_response and health_response.get("status") == "healthy" else "degraded"
        except:
            system_status = "degraded"

        return SystemCapabilities(
            supported_modes=list(QUERY_MODE_CONFIGS.values()),
            current_load=current_load,
            estimated_response_times=estimated_times,
            feature_flags={
                "enhanced_queries": True,
                "structured_analysis": True,
                "multi_perspective": True,
                "user_quotes": True,
                "query_validation": True,
                "validation_workflows": True,
                "user_guided_validation": True,
                "unified_system": True
            },
            system_status=system_status,
            validation_supported=True,
            validation_types=[ValidationType.BASIC, ValidationType.COMPREHENSIVE, ValidationType.USER_GUIDED]
        )

    except Exception as e:
        logger.error(f"Error getting system capabilities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting system capabilities: {str(e)}")


@router.post("/validate", response_model=QueryValidationResult)
async def validate_query_for_modes(request: EnhancedQueryRequest) -> QueryValidationResult:
    """Validate a query for mode compatibility and suggest validation type."""
    try:
        query_text = request.query.lower()

        # Mode compatibility scoring (existing logic)
        compatibility_scores = {}

        facts_keywords = ["什么", "多少", "哪个", "规格", "参数", "尺寸", "重量"]
        facts_score = sum(1 for kw in facts_keywords if kw in query_text) / len(facts_keywords)
        compatibility_scores[QueryMode.FACTS] = facts_score > 0.2

        features_keywords = ["应该", "值得", "建议", "增加", "功能", "是否"]
        features_score = sum(1 for kw in features_keywords if kw in query_text) / len(features_keywords)
        compatibility_scores[QueryMode.FEATURES] = features_score > 0.2

        tradeoffs_keywords = ["vs", "对比", "优缺点", "利弊", "比较", "选择"]
        tradeoffs_score = sum(1 for kw in tradeoffs_keywords if kw in query_text) / len(tradeoffs_keywords)
        compatibility_scores[QueryMode.TRADEOFFS] = tradeoffs_score > 0.2

        scenarios_keywords = ["场景", "情况下", "使用时", "体验", "用户", "日常"]
        scenarios_score = sum(1 for kw in scenarios_keywords if kw in query_text) / len(scenarios_keywords)
        compatibility_scores[QueryMode.SCENARIOS] = scenarios_score > 0.2

        debate_keywords = ["观点", "看法", "认为", "角度", "团队", "讨论"]
        debate_score = sum(1 for kw in debate_keywords if kw in query_text) / len(debate_keywords)
        compatibility_scores[QueryMode.DEBATE] = debate_score > 0.2

        quotes_keywords = ["评价", "反馈", "评论", "用户说", "真实", "体验"]
        quotes_score = sum(1 for kw in quotes_keywords if kw in query_text) / len(quotes_keywords)
        compatibility_scores[QueryMode.QUOTES] = quotes_score > 0.2

        mode_scores = {
            QueryMode.FACTS: max(facts_score, 0.4),
            QueryMode.FEATURES: features_score,
            QueryMode.TRADEOFFS: tradeoffs_score,
            QueryMode.SCENARIOS: scenarios_score,
            QueryMode.DEBATE: debate_score,
            QueryMode.QUOTES: quotes_score
        }

        suggested_mode = max(mode_scores.items(), key=lambda x: x[1])[0]
        confidence = max(mode_scores.values())

        # Validation recommendations
        validation_recommended = False
        suggested_validation_type = None

        if confidence < 0.6:
            validation_recommended = True
            suggested_validation_type = ValidationType.COMPREHENSIVE
        elif any(score > 0.3 for score in [features_score, tradeoffs_score, scenarios_score]):
            validation_recommended = True
            suggested_validation_type = ValidationType.USER_GUIDED
        elif facts_score > 0.6:
            suggested_validation_type = ValidationType.BASIC

        recommendations = []
        if confidence < 0.4:
            recommendations.append("建议使用车辆规格查询模式（默认模式）")
        if validation_recommended:
            recommendations.append(f"建议启用{suggested_validation_type.value}验证以提高答案质量")
        if facts_score > 0.3:
            recommendations.append("查询包含具体规格问题，车辆规格查询模式最适合")

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


# Legacy endpoints (unchanged)
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
    model_map = {
        "Toyota": ["Camry", "Corolla", "RAV4", "Highlander", "Tacoma"],
        "Honda": ["Civic", "Accord", "CR-V", "Pilot", "Odyssey"],
        "Ford": ["Mustang", "F-150", "Escape", "Explorer", "Edge"],
        "BMW": ["3 Series", "5 Series", "X3", "X5", "i4"]
    }
    return model_map.get(manufacturer, ["Model S", "Model 3", "Model X", "Model Y", "Cybertruck"])


@router.get("/queue-status", response_model=Dict[str, Any])
async def get_queue_status() -> Dict[str, Any]:
    """Get status of the job chain queue system."""
    try:
        return job_chain.get_queue_status()
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting queue status: {str(e)}")


@router.post("/debug-retrieval", response_model=Dict[str, Any])
async def debug_document_retrieval(request: Dict[str, str]) -> Dict[str, Any]:
    """Debug endpoint to retrieve documents from vector store for browsing."""
    try:
        from src.core.background.models import get_vector_store

        query = request.get("query", "document")
        limit = min(int(request.get("limit", 100)), 500)

        vector_store = get_vector_store()
        results = vector_store.similarity_search_with_score(
            query=query,
            k=limit,
            metadata_filter=None
        )

        documents = []
        for doc, score in results:
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
                "content": doc.page_content,
                "metadata": cleaned_metadata,
                "relevance_score": float(score) if isinstance(score, (np.floating, np.float32, np.float64)) else score
            })

        processing_stats = calculate_processing_statistics(documents)

        return {
            "documents": documents,
            "total_found": len(documents),
            "query_used": query,
            "processing_stats": processing_stats
        }

    except Exception as e:
        logger.error(f"Debug retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document retrieval failed: {str(e)}")