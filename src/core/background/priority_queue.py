"""
Priority queue management system for GPU resources.

This module provides a priority queue system for managing tasks that
need to access limited GPU resources, ensuring important tasks get
access to the GPU before less important ones.
"""

import json
import time
import logging
from typing import Dict, Any, Optional

from .common import check_gpu_health, get_redis_client

logger = logging.getLogger(__name__)


class PriorityQueueManager:
    """Manages task priorities across different queues."""

    def __init__(self, redis_client=None, priority_levels=None):
        """Initialize the priority queue manager."""
        if redis_client is None:
            self.redis = get_redis_client()
        else:
            self.redis = redis_client

        self.priority_queue_key = "priority_task_queue"
        self.active_task_key = "active_gpu_task"

        # Set priority levels - lower number = higher priority
        self.priority_levels = priority_levels or {
            "inference_tasks": 1,  # Highest priority
            "reranking_tasks": 2,
            "embedding_tasks": 3,
            "transcription_tasks": 4
        }

    def register_task(self, queue_name: str, task_id: str, metadata: Dict[str, Any] = None) -> str:
        """Register a task in the priority system."""
        priority = self.priority_levels.get(queue_name, 999)
        task_data = {
            "task_id": task_id,
            "queue_name": queue_name,
            "priority": priority,
            "registered_at": time.time(),
            "metadata": metadata or {}
        }

        # Add to sorted set with priority as score
        self.redis.zadd(self.priority_queue_key, {json.dumps(task_data): priority})
        logger.info(f"Registered task {task_id} from queue {queue_name} with priority {priority}")
        return task_id

    def get_active_task(self) -> Optional[Dict[str, Any]]:
        """Get the currently active GPU task, if any."""
        active_task = self.redis.get(self.active_task_key)
        if active_task:
            return json.loads(active_task)
        return None

    def mark_task_active(self, task_data: Dict[str, Any]) -> None:
        """Mark a task as currently active on GPU."""
        self.redis.set(self.active_task_key, json.dumps(task_data), ex=3600)  # 1 hour expiry as safety
        logger.info(f"Marked task {task_data.get('task_id')} as active on GPU")

    def mark_task_completed(self, task_id: str) -> None:
        """Mark a task as completed and remove from priority queue."""
        # Remove from active task if it's this task
        active_task = self.get_active_task()
        if active_task and active_task.get("task_id") == task_id:
            self.redis.delete(self.active_task_key)
            logger.info(f"Removed task {task_id} from active GPU task")

        # Remove from priority queue by searching for task with this ID
        all_tasks = self.redis.zrange(self.priority_queue_key, 0, -1, withscores=True)
        for task_json, _ in all_tasks:
            task = json.loads(task_json)
            if task.get("task_id") == task_id:
                self.redis.zrem(self.priority_queue_key, task_json)
                logger.info(f"Removed task {task_id} from priority queue")
                break

    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get the highest priority task that should run next."""
        # Get the task with the lowest score (highest priority)
        tasks = self.redis.zrange(self.priority_queue_key, 0, 0, withscores=True)
        if not tasks:
            return None

        task_json, priority = tasks[0]
        return json.loads(task_json)

    def can_run_task(self, queue_name: str, task_id: str) -> bool:
        """Determine if a task can run according to priorities, with anti-starvation measures."""
        # Get the task details
        task_details = None
        all_tasks = self.redis.zrange(self.priority_queue_key, 0, -1, withscores=True)
        for task_json, _ in all_tasks:
            task = json.loads(task_json)
            if task.get("task_id") == task_id:
                task_details = task
                break

        if not task_details:
            # Task not found in queue
            return False

        # For inference tasks, always allow them to run immediately
        # This ensures they have absolute priority
        if queue_name == "inference_tasks":
            active_task = self.get_active_task()
            # If no active task, or active task is not inference, allow to run
            if not active_task or active_task.get("queue_name") != "inference_tasks":
                return True

        # Check if task has been waiting too long (5 minutes)
        current_time = time.time()
        registered_time = task_details.get("registered_at", current_time)

        if current_time - registered_time > 300:  # 5 minutes
            # Task has waited too long, allow it to run regardless of priority
            logger.info(f"Task {task_id} has waited over 5 minutes, allowing to run")
            return True

        # If no active task, can run if it's the highest priority task
        active_task = self.get_active_task()
        if not active_task:
            # Check if this is the highest priority task
            next_task = self.get_next_task()

            # If no tasks in queue or this is the highest priority task, run it
            if not next_task or next_task.get("task_id") == task_id:
                return True

            # If this task's queue has higher priority than next task, run it
            task_priority = self.priority_levels.get(queue_name, 999)
            next_priority = next_task.get("priority", 999)
            return task_priority <= next_priority

        # If there's an active task of higher priority, wait
        active_priority = active_task.get("priority", 999)
        task_priority = self.priority_levels.get(queue_name, 999)

        # Only allow if this task has strictly higher priority than the running task
        return task_priority < active_priority

    def get_queue_status(self) -> Dict[str, Any]:
        """Get the status of the priority queue and all tasks."""
        # Get active task
        active_task_json = self.redis.get(self.active_task_key)
        active_task = json.loads(active_task_json) if active_task_json else None

        # Get all tasks in priority queue
        all_tasks = self.redis.zrange(
            self.priority_queue_key, 0, -1,
            withscores=True
        )

        # Process tasks
        priority_tasks = []
        tasks_by_priority = {}
        tasks_by_queue = {}

        for task_json, priority in all_tasks:
            task = json.loads(task_json)
            task["priority"] = priority
            priority_tasks.append(task)

            # Count by priority
            if priority not in tasks_by_priority:
                tasks_by_priority[priority] = 0
            tasks_by_priority[priority] += 1

            # Count by queue
            queue_name = task.get("queue_name", "unknown")
            if queue_name not in tasks_by_queue:
                tasks_by_queue[queue_name] = 0
            tasks_by_queue[queue_name] += 1

        # Sort by priority (low to high = high to low priority)
        priority_tasks.sort(key=lambda x: x["priority"])

        return {
            "active_task": active_task,
            "priority_tasks": priority_tasks,
            "tasks_by_priority": tasks_by_priority,
            "tasks_by_queue": tasks_by_queue,
            "priority_levels": self.priority_levels,
            "total_tasks": len(priority_tasks),
            "timestamp": time.time()
        }

    def check_health(self) -> Dict[str, Any]:
        """Check the health of the priority queue system."""
        result = {
            "status": "healthy",
            "issues": [],
            "recommendations": []
        }

        # Check for inconsistencies in the priority queue
        active_task = self.get_active_task()

        # Check if there's an active task but no GPU worker is running
        if active_task:
            # Check how long the task has been active
            task_age = time.time() - active_task.get("registered_at", time.time())

            # If task has been active too long, check if the worker is still running
            if task_age > 900:  # 15 minutes
                task_id = active_task.get("task_id")
                job_id = active_task.get("job_id")

                # Check if GPU is healthy
                is_healthy, status_message = check_gpu_health()

                if not is_healthy:
                    # GPU is unhealthy but there's an active task - flag for reset
                    result["status"] = "unhealthy"
                    result["issues"].append(f"GPU unhealthy but task {task_id} is marked active for {task_age:.2f}s")
                    result["recommendations"].append("Reset active GPU task")

        # Check if there are tasks in queue but no active task
        queue_tasks = self.redis.zcard(self.priority_queue_key)
        if not active_task and queue_tasks > 0:
            result["issues"].append(f"Priority queue has {queue_tasks} tasks but no active task")
            result["recommendations"].append("Check if GPU workers are running")

        return result

    def balance_queues(self) -> Dict[str, Any]:
        """
        Balance workload across queues to prevent starvation.
        Temporarily boosts priority of lower-priority queues if they have many tasks waiting.
        """
        result = {
            "adjusted_priorities": [],
            "restored_priorities": []
        }

        # Get queue lengths for each priority level
        queue_lengths = {}
        for queue_name in self.priority_levels.keys():
            # Count tasks in this queue
            count = 0
            all_tasks = self.redis.zrange(self.priority_queue_key, 0, -1, withscores=True)
            for task_json, _ in all_tasks:
                task = json.loads(task_json)
                if task.get("queue_name") == queue_name:
                    count += 1
            queue_lengths[queue_name] = count

        # Check for boosted priorities that should be restored
        priority_boost_keys = self.redis.keys("priority_boost:*")
        for key in priority_boost_keys:
            boost_data_json = self.redis.get(key)
            queue_name = key.split(":")[-1]

            # If the key exists but data is invalid or missing, delete it
            if not boost_data_json:
                self.redis.delete(key)
                continue

            try:
                boost_data = json.loads(boost_data_json)
                # Check if the boost has expired
                if time.time() > boost_data.get("expires_at", 0):
                    # Restore original priority
                    self.priority_levels[queue_name] = boost_data.get("original", 4)
                    result["restored_priorities"].append(queue_name)
                    self.redis.delete(key)
                    logger.info(f"Restored original priority for {queue_name}")
            except:
                # Invalid data, delete the key
                self.redis.delete(key)

        # Check for queue imbalances
        high_priority_queues = ["inference_tasks", "reranking_tasks"]
        low_priority_queues = ["embedding_tasks", "transcription_tasks"]

        # If lower priority queues have too many tasks waiting,
        # temporarily boost their priority to prevent starvation
        if all(queue_lengths.get(q, 0) == 0 for q in high_priority_queues) and \
                any(queue_lengths.get(q, 0) > 10 for q in low_priority_queues):

            # Example: If transcription tasks are piling up, adjust their priority
            if queue_lengths.get("transcription_tasks", 0) > 10:
                original_priority = self.priority_levels.get("transcription_tasks", 4)
                boosted_priority = max(original_priority - 2, 2)  # Don't go higher than 2

                # Record the temporary priority boost
                self.redis.set(
                    "priority_boost:transcription_tasks",
                    json.dumps({
                        "original": original_priority,
                        "boosted": boosted_priority,
                        "expires_at": time.time() + 600  # 10 minute expiry
                    }),
                    ex=600  # 10 minute expiry
                )

                # Apply the boost
                self.priority_levels["transcription_tasks"] = boosted_priority
                result["adjusted_priorities"].append("transcription_tasks")
                logger.info(
                    f"Temporarily boosted transcription_tasks priority from {original_priority} to {boosted_priority} for 10 minutes")

        return result

    def boost_job_priority(self, job_id: str) -> Dict[str, Any]:
        """
        Boost the priority of all tasks related to a specific job.

        Args:
            job_id: ID of the job to boost

        Returns:
            Dictionary with boosting results
        """
        # Get all tasks in the priority queue
        all_tasks = self.redis.zrange(
            self.priority_queue_key, 0, -1,
            withscores=True
        )

        boosted_tasks = 0

        for task_json, priority in all_tasks:
            task = json.loads(task_json)

            # Check if this task belongs to the job
            if task.get("job_id") == job_id or task.get("metadata", {}).get("parent_job_id") == job_id:
                # Remove the task from the queue
                self.redis.zrem(self.priority_queue_key, task_json)

                # Check current minimum priority (highest priority task)
                min_priority = self.redis.zrange(
                    self.priority_queue_key, 0, 0,
                    withscores=True,
                    desc=False
                )

                # Calculate new boosted priority
                new_priority = 0  # Default to highest priority
                if min_priority:
                    # Set priority just below the highest priority task
                    new_priority = min_priority[0][1] - 0.1
                    if new_priority < 0:
                        new_priority = 0

                # Update task priority
                task["priority"] = new_priority

                # Add back to queue with boosted priority
                self.redis.zadd(self.priority_queue_key, {json.dumps(task): new_priority})

                boosted_tasks += 1

                logger.info(f"Boosted task {task.get('task_id')} for job {job_id} to priority {new_priority}")

        return {
            "job_id": job_id,
            "boosted_tasks": boosted_tasks
        }

# Initialize the global priority queue instance
priority_queue = PriorityQueueManager()