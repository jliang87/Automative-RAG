#!/usr/bin/env python3
"""
Priority Queue Monitor

This script monitors the priority queue system and provides visibility into
which tasks are waiting and running.
"""

import os
import time
import json
import redis
from datetime import datetime
from tabulate import tabulate

# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)

# Connect to Redis
redis_kwargs = {
    "host": REDIS_HOST,
    "port": REDIS_PORT,
    "decode_responses": True,
}
if REDIS_PASSWORD:
    redis_kwargs["password"] = REDIS_PASSWORD

redis_client = redis.Redis(**redis_kwargs)


def get_queue_status():
    """Get status of the priority queue and active task."""
    # Get active task
    active_task_json = redis_client.get("active_gpu_task")
    active_task = json.loads(active_task_json) if active_task_json else None

    # Get all tasks in priority queue
    all_tasks = redis_client.zrange(
        "priority_task_queue", 0, -1,
        withscores=True
    )

    priority_tasks = []
    for task_json, priority in all_tasks:
        task = json.loads(task_json)
        task["priority"] = priority
        priority_tasks.append(task)

    # Sort by priority (low to high = high to low priority)
    priority_tasks.sort(key=lambda x: x["priority"])

    return {
        "active_task": active_task,
        "priority_tasks": priority_tasks
    }


def format_time(timestamp):
    """Format a timestamp to a readable string."""
    if not timestamp:
        return "N/A"

    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(timestamp):
    """Format duration in seconds to a readable string."""
    if not timestamp:
        return "N/A"

    seconds = time.time() - timestamp

    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def display_queue_status():
    """Display the queue status in a table format."""
    status = get_queue_status()

    # Display active task
    print("\n=== ACTIVE GPU TASK ===")
    active_task = status["active_task"]
    if active_task:
        active_data = [
            ["Task ID", active_task.get("task_id", "N/A")],
            ["Queue", active_task.get("queue_name", "N/A")],
            ["Priority", active_task.get("priority", "N/A")],
            ["Job ID", active_task.get("job_id", "N/A")],
            ["Active Since", format_duration(active_task.get("registered_at"))]
        ]
        print(tabulate(active_data, tablefmt="grid"))
    else:
        print("No active GPU task")

    # Display waiting tasks
    print("\n=== WAITING TASKS ===")
    priority_tasks = status["priority_tasks"]
    if priority_tasks:
        task_rows = []
        for task in priority_tasks:
            waiting_time = format_duration(task.get("registered_at"))
            task_rows.append([
                task.get("task_id", "N/A")[:12] + "...",
                task.get("queue_name", "N/A"),
                task.get("priority", "N/A"),
                task.get("metadata", {}).get("job_id", "N/A")[:8] + "...",
                waiting_time
            ])

        headers = ["Task ID", "Queue", "Priority", "Job ID", "Waiting Time"]
        print(tabulate(task_rows, headers=headers, tablefmt="grid"))
    else:
        print("No tasks in waiting queue")


def main():
    """Main function to continuously monitor the queue."""
    try:
        while True:
            os.system('clear')  # Clear screen
            print("=== PRIORITY QUEUE MONITOR ===")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Display queue status
            display_queue_status()

            # Wait before refreshing
            print("\nPress Ctrl+C to exit. Refreshing in 2 seconds...")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nExiting monitor.")


if __name__ == "__main__":
    main()