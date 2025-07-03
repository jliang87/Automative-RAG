"""
Queue Manager - Extracted from JobChain
Handles queue state management and task queuing
"""

import json
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class QueueManager:
    """
    Manages queue state and task queuing.

    Extracted from the massive JobChain to provide focused queue management.
    """

    def __init__(self):
        from core.background.common import get_redis_client
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
        from core.orchestration.job_tracker import job_tracker
        job_tracker.update_job_progress(job_id, None, f"Waiting for {queue_name} to become available")

    def process_waiting_tasks(self, queue_name: str) -> None:
        """Process any tasks waiting for this queue to become free."""
        waiting_task_json = self.redis.rpop(f"waiting_tasks:{queue_name}")
        if waiting_task_json:
            waiting_task = json.loads(waiting_task_json)
            logger.info(f"Processing waiting task for queue {queue_name}: {waiting_task['task_name']}")

            # Execute the waiting task immediately
            from core.orchestration.job_chain import job_chain
            job_chain._execute_task_immediately(
                waiting_task["job_id"],
                waiting_task["task_name"],
                waiting_task["queue_name"],
                waiting_task["data"]
            )

    def get_queue_status(self) -> Dict[str, Any]:
        """Get the status of all queues."""
        queue_names = ["cpu_tasks", "transcription_tasks", "embedding_tasks", "inference_tasks"]
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
        """Get detailed queue statistics."""
        queue_names = ["cpu_tasks", "transcription_tasks", "embedding_tasks", "inference_tasks"]
        stats = {
            "total_queues": len(queue_names),
            "busy_queues": 0,
            "total_waiting_tasks": 0,
            "queue_details": {}
        }

        for queue_name in queue_names:
            waiting_count = self.redis.llen(f"waiting_tasks:{queue_name}")
            is_busy = self.is_queue_busy(queue_name)

            if is_busy:
                stats["busy_queues"] += 1

            stats["total_waiting_tasks"] += waiting_count
            stats["queue_details"][queue_name] = {
                "busy": is_busy,
                "waiting_tasks": waiting_count
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
        """Clear all queue state (for system reset/debugging)."""
        queue_names = ["cpu_tasks", "transcription_tasks", "embedding_tasks", "inference_tasks"]
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


# Global queue manager instance
queue_manager = QueueManager()