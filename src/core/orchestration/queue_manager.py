import json
import time
import logging
from typing import Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class QueueNames(Enum):
    """
    ✅ CORRECTED: Queue names based on Tesla T4 memory constraints.

    ⚠️  CRITICAL DESIGN CONSTRAINTS:
    - Tesla T4 has 16GB GPU memory
    - GPU queues fit specific models: transcription, embedding, LLM
    - All LLM tasks (inference + validation) MUST use same queue to avoid memory conflicts
    - CPU queues limited by available cores/memory
    - ONE task per queue ensures atomic memory usage
    """

    # CPU-based queues (limited by CPU cores/memory)
    CPU_TASKS = "cpu_tasks"  # All CPU work: knowledge validation, meta-validation, auto-fetch, etc.

    # GPU-based queues (limited by Tesla T4 16GB memory)
    TRANSCRIPTION_TASKS = "transcription_tasks"  # Whisper model
    EMBEDDING_TASKS = "embedding_tasks"          # Sentence transformer models
    LLM_TASKS = "llm_tasks"                      # ✅ RENAMED: All LLM work (inference + validation)

    @classmethod
    def get_all_queue_names(cls) -> List[str]:
        """Get list of all queue names."""
        return [queue.value for queue in cls]

    @classmethod
    def get_gpu_queues(cls) -> List[str]:
        """Get GPU-constrained queues."""
        return [
            cls.TRANSCRIPTION_TASKS.value,
            cls.EMBEDDING_TASKS.value,
            cls.LLM_TASKS.value  # ✅ RENAMED from inference_tasks
        ]

    @classmethod
    def get_cpu_queues(cls) -> List[str]:
        """Get CPU-constrained queues."""
        return [cls.CPU_TASKS.value]


class QueueManager:
    """
    Manages queue state and task queuing.
    ⚠️  CORRECTED: Respects Tesla T4 memory constraints.
    """

    def __init__(self):
        from src.core.background.common import get_redis_client
        self.redis = get_redis_client()

    def is_queue_busy(self, queue_name: str) -> bool:
        """Check if a queue is currently busy."""
        return self.redis.exists(f"queue_busy:{queue_name}")

    def mark_queue_busy(self, queue_name: str, job_id: str, task_name: str) -> None:
        """Mark a queue as busy."""
        busy_info = {
            "job_id": job_id,
            "task_name": task_name,
            "started_at": time.time()
        }
        self.redis.set(f"queue_busy:{queue_name}", json.dumps(busy_info, ensure_ascii=False), ex=3600)
        logger.info(f"Marked queue {queue_name} as busy for job {job_id}")

    def mark_queue_free(self, queue_name: str) -> None:
        """Mark a queue as free."""
        self.redis.delete(f"queue_busy:{queue_name}")
        logger.info(f"Marked queue {queue_name} as free")

    def queue_task(self, job_id: str, task_name: str, queue_name: str, data: Dict[str, Any]) -> None:
        """Queue a task to wait for the queue to become available."""
        queued_task = {
            "job_id": job_id,
            "task_name": task_name,
            "queue_name": queue_name,
            "data": data,
            "queued_at": time.time()
        }

        # Add to waiting queue
        self.redis.lpush(f"waiting_tasks:{queue_name}", json.dumps(queued_task, ensure_ascii=False))
        logger.info(f"Queued task {task_name} for job {job_id} in {queue_name}")

        # Update job progress to show waiting
        from src.core.orchestration.job_tracker import job_tracker
        job_tracker.update_job_progress(job_id, None, f"Waiting for {queue_name} to become available")

    def process_waiting_tasks(self, queue_name: str) -> None:
        """Process any tasks waiting for this queue to become free."""
        waiting_task_json = self.redis.rpop(f"waiting_tasks:{queue_name}")
        if waiting_task_json:
            waiting_task = json.loads(waiting_task_json)
            logger.info(f"Processing waiting task for queue {queue_name}: {waiting_task['task_name']}")

            # Execute the waiting task immediately
            from src.core.orchestration.job_chain import job_chain
            job_chain._execute_task_immediately(
                waiting_task["job_id"],
                waiting_task["task_name"],
                waiting_task["queue_name"],
                waiting_task["data"]
            )

    def get_queue_status(self) -> Dict[str, Any]:
        """✅ CORRECTED: Get status using Tesla T4 constrained queue names."""
        queue_names = QueueNames.get_all_queue_names()
        queue_status = {}

        for queue_name in queue_names:
            busy_info_json = self.redis.get(f"queue_busy:{queue_name}")
            waiting_count = self.redis.llen(f"waiting_tasks:{queue_name}")

            if busy_info_json:
                busy_info = json.loads(busy_info_json)
                queue_status[queue_name] = {
                    "status": "busy",
                    "current_job": busy_info["job_id"],
                    "current_task": busy_info["task_name"],
                    "busy_since": busy_info["started_at"],
                    "waiting_tasks": waiting_count
                }
            else:
                queue_status[queue_name] = {
                    "status": "free",
                    "waiting_tasks": waiting_count
                }

        return queue_status

    def get_queue_statistics(self) -> Dict[str, Any]:
        """✅ CORRECTED: Get statistics respecting hardware constraints."""
        queue_names = QueueNames.get_all_queue_names()
        stats = {
            "total_queues": len(queue_names),
            "busy_queues": 0,
            "total_waiting_tasks": 0,
            "queue_details": {},
            "hardware_constraints": {
                "gpu_model": "Tesla T4",
                "gpu_memory_gb": 16,
                "gpu_queues": QueueNames.get_gpu_queues(),
                "cpu_queues": QueueNames.get_cpu_queues(),
                "constraint_note": "Queue design optimized for Tesla T4 memory atomicity"
            }
        }

        for queue_name in queue_names:
            waiting_count = self.redis.llen(f"waiting_tasks:{queue_name}")
            is_busy = self.is_queue_busy(queue_name)

            if is_busy:
                stats["busy_queues"] += 1

            stats["total_waiting_tasks"] += waiting_count
            stats["queue_details"][queue_name] = {
                "busy": is_busy,
                "waiting_tasks": waiting_count,
                "queue_type": "GPU" if queue_name in QueueNames.get_gpu_queues() else "CPU"
            }

        stats["free_queues"] = stats["total_queues"] - stats["busy_queues"]
        return stats

    def clear_queue_state(self, queue_name: str) -> Dict[str, Any]:
        """Clear all state for a specific queue (for debugging/maintenance)."""
        result = {
            "queue_name": queue_name,
            "busy_state_cleared": False,
            "waiting_tasks_cleared": 0
        }

        # Clear busy state
        if self.redis.delete(f"queue_busy:{queue_name}"):
            result["busy_state_cleared"] = True
            logger.info(f"Cleared busy state for queue {queue_name}")

        # Clear waiting tasks
        waiting_count = self.redis.llen(f"waiting_tasks:{queue_name}")
        if waiting_count > 0:
            self.redis.delete(f"waiting_tasks:{queue_name}")
            result["waiting_tasks_cleared"] = waiting_count
            logger.info(f"Cleared {waiting_count} waiting tasks for queue {queue_name}")

        return result

    def clear_all_queue_state(self) -> Dict[str, Any]:
        """✅ CORRECTED: Clear all queue state respecting hardware constraints."""
        queue_names = QueueNames.get_all_queue_names()
        results = {
            "total_queues_cleared": 0,
            "total_waiting_tasks_cleared": 0,
            "queue_results": {}
        }

        for queue_name in queue_names:
            queue_result = self.clear_queue_state(queue_name)
            results["queue_results"][queue_name] = queue_result
            results["total_queues_cleared"] += 1
            results["total_waiting_tasks_cleared"] += queue_result["waiting_tasks_cleared"]

        logger.info(
            f"Cleared all queue state: {results['total_queues_cleared']} queues, {results['total_waiting_tasks_cleared']} waiting tasks")
        return results

    def validate_queue_name(self, queue_name: str) -> bool:
        """Validate that a queue name is in the hardware-constrained list."""
        return queue_name in QueueNames.get_all_queue_names()

    def get_queue_configuration(self) -> Dict[str, Dict[str, Any]]:
        """✅ CORRECTED: Get Dramatiq configuration respecting hardware constraints."""
        base_config = {
            "max_retries": 3,
            "min_backoff": 15000,
            "max_backoff": 300000,
            "store_results": True
        }

        # Configurations optimized for Tesla T4 constraints
        queue_configs = {
            # CPU-constrained queue
            QueueNames.CPU_TASKS.value: {
                **base_config,
                "max_retries": 3,
                "min_backoff": 15000,
                "note": "All CPU work: knowledge validation, meta-validation, auto-fetch"
            },

            # GPU-constrained queues (Tesla T4 optimized)
            QueueNames.TRANSCRIPTION_TASKS.value: {
                **base_config,
                "max_retries": 2,
                "min_backoff": 60000,
                "max_backoff": 900000,
                "note": "Whisper model - Tesla T4 optimized"
            },
            QueueNames.EMBEDDING_TASKS.value: {
                **base_config,
                "max_retries": 3,
                "min_backoff": 10000,
                "max_backoff": 180000,
                "note": "Sentence transformer models - Tesla T4 optimized"
            },
            QueueNames.LLM_TASKS.value: {  # ✅ RENAMED from inference_tasks
                **base_config,
                "max_retries": 2,
                "min_backoff": 30000,
                "max_backoff": 600000,
                "note": "All LLM work (inference + validation) - shares DeepSeq model memory"
            }
        }

        return queue_configs

    def get_recommended_queue_for_task_type(self, task_type: str) -> str:
        """✅ CORRECTED: Get queue mapping respecting Tesla T4 constraints."""
        task_to_queue_mapping = {
            # CPU tasks (all go to single CPU queue)
            "download_video": QueueNames.CPU_TASKS.value,
            "process_pdf": QueueNames.CPU_TASKS.value,
            "process_text": QueueNames.CPU_TASKS.value,
            "knowledge_validation": QueueNames.CPU_TASKS.value,     # ✅ CORRECTED
            "meta_validation": QueueNames.CPU_TASKS.value,          # ✅ CORRECTED
            "auto_fetch": QueueNames.CPU_TASKS.value,               # ✅ CORRECTED

            # GPU tasks (Tesla T4 constrained)
            "transcribe_video": QueueNames.TRANSCRIPTION_TASKS.value,
            "generate_embeddings": QueueNames.EMBEDDING_TASKS.value,
            "retrieve_documents": QueueNames.EMBEDDING_TASKS.value,

            # ✅ CRITICAL: All LLM tasks use same queue to share model memory
            "llm_inference": QueueNames.LLM_TASKS.value,            # ✅ RENAMED queue
            "pre_llm_validation": QueueNames.LLM_TASKS.value,       # ✅ CORRECTED - same queue
            "post_llm_validation": QueueNames.LLM_TASKS.value,      # ✅ CORRECTED - same queue
            "final_validation": QueueNames.LLM_TASKS.value          # ✅ CORRECTED - same queue
        }

        return task_to_queue_mapping.get(task_type, QueueNames.CPU_TASKS.value)

    def get_hardware_constraints_info(self) -> Dict[str, Any]:
        """✅ NEW: Document hardware constraints for the queue design."""
        return {
            "gpu_constraints": {
                "model": "Tesla T4",
                "memory_gb": 16,
                "design_principle": "One task per queue ensures atomic memory usage",
                "memory_allocation": {
                    "transcription": "Whisper model ~2-4GB",
                    "embedding": "Sentence transformers ~1-2GB",
                    "llm": "DeepSeq model ~8-12GB (shared by inference + validation)"
                }
            },
            "cpu_constraints": {
                "design_principle": "All CPU work uses single queue to manage core/memory limits",
                "task_types": [
                    "knowledge_validation",
                    "meta_validation",
                    "auto_fetch",
                    "document_processing",
                    "video_download"
                ]
            },
            "queue_atomicity": {
                "principle": "One task per queue prevents memory conflicts",
                "critical_rule": "Never create new GPU queues - breaks memory model",
                "llm_sharing": "All LLM tasks (inference + validation) share same queue/model"
            }
        }


# Global queue manager instance
queue_manager = QueueManager()