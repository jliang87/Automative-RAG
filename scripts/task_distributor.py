#!/usr/bin/env python3
"""
Task Distributor

This script provides utilities for managing tasks across the various workers.
It can be used to submit test tasks, clear queues, and monitor system health.
"""

import os
import sys
import time
import json
import uuid
import redis
import argparse
import requests
from tabulate import tabulate

# Redis configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)
API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "default-api-key")

# Connect to Redis
redis_kwargs = {
    "host": REDIS_HOST,
    "port": REDIS_PORT,
    "decode_responses": True,
}
if REDIS_PASSWORD:
    redis_kwargs["password"] = REDIS_PASSWORD

redis_client = redis.Redis(**redis_kwargs)


def get_queue_lengths():
    """Get the number of tasks in each Dramatiq queue."""
    all_queues = []

    # Scan for queue keys with wildcard pattern
    cursor = 0
    while True:
        cursor, keys = redis_client.scan(cursor, "dramatiq:*:msgs", 100)
        all_queues.extend(keys)
        if cursor == 0:
            break

    # Get queue lengths
    queue_lengths = {}
    for queue_key in all_queues:
        queue_name = queue_key.split(":")[1]
        queue_lengths[queue_name] = redis_client.llen(queue_key)

    return queue_lengths


def clear_queues(queue_names=None):
    """Clear specified queues or all queues if none specified."""
    if queue_names is None:
        # Get all queue names
        queue_lengths = get_queue_lengths()
        queue_names = list(queue_lengths.keys())

    for queue_name in queue_names:
        queue_key = f"dramatiq:{queue_name}:msgs"
        queue_length = redis_client.llen(queue_key)

        # Delete the queue
        redis_client.delete(queue_key)
        print(f"Cleared {queue_length} messages from queue '{queue_name}'")


def reset_priority_system():
    """Reset the priority queue system."""
    # Delete priority queue and active task
    priority_count = redis_client.zcard("priority_task_queue")
    redis_client.delete("priority_task_queue")
    redis_client.delete("active_gpu_task")

    print(f"Reset priority queue system: removed {priority_count} tasks and cleared active task")


def submit_test_task(task_type, count=1):
    """Submit a test task to the system."""
    headers = {"x-token": API_KEY}

    for i in range(count):
        if task_type == "inference":
            # Submit a test query
            data = {
                "query": f"Test query {uuid.uuid4()}",
                "metadata_filter": None
            }
            response = requests.post(f"{API_URL}/query", json=data, headers=headers)

        elif task_type == "embedding":
            # Submit a test text for embedding
            data = {
                "content": f"Test content for embedding {uuid.uuid4()}",
                "metadata": {
                    "source": "manual",
                    "title": f"Test Document {i + 1}"
                }
            }
            response = requests.post(f"{API_URL}/ingest/text", json=data, headers=headers)

        elif task_type == "whisper":
            # This requires an actual audio file, so we just print instructions
            print("To test Whisper, you need to upload an audio file via the API.")
            print(
                "Use: curl -X POST -F 'file=@audio.mp3' -H 'x-token: your-api-key' http://localhost:8000/ingest/audio")
            return

        elif task_type == "pdf":
            # This requires an actual PDF file, so we just print instructions
            print("To test PDF processing, you need to upload a PDF file via the API.")
            print(
                "Use: curl -X POST -F 'file=@document.pdf' -H 'x-token: your-api-key' http://localhost:8000/ingest/pdf")
            return

        # Print response
        if response.status_code == 200:
            result = response.json()
            print(f"Submitted test {task_type} task {i + 1}/{count}: Job ID {result.get('job_id', 'N/A')}")
        else:
            print(f"Failed to submit test {task_type} task: {response.text}")


def display_system_status():
    """Display overall system status."""
    # Get queue lengths
    queue_lengths = get_queue_lengths()

    # Get job statistics
    job_stats = count_jobs_by_status()

    # Get priority queue status
    priority_queue_count = redis_client.zcard("priority_task_queue")
    active_task = redis_client.get("active_gpu_task")
    active_task_info = json.loads(active_task) if active_task else None

    # Display information in tables
    print("\n=== QUEUE STATUS ===")
    queue_rows = []
    for queue_name, length in queue_lengths.items():
        queue_rows.append([queue_name, length])
    print(tabulate(queue_rows, headers=["Queue", "Pending Tasks"], tablefmt="grid"))

    print("\n=== JOB STATUS ===")
    job_rows = []
    for status, count in job_stats.items():
        job_rows.append([status, count])
    print(tabulate(job_rows, headers=["Status", "Count"], tablefmt="grid"))

    print("\n=== PRIORITY SYSTEM ===")
    priority_data = [
        ["Tasks in Priority Queue", priority_queue_count],
        ["Active Task", active_task_info["task_id"] if active_task_info else "None"]
    ]
    print(tabulate(priority_data, tablefmt="grid"))


def count_jobs_by_status():
    """Count jobs by status."""
    job_key = "rag_system:jobs"
    all_jobs = redis_client.hgetall(job_key)

    status_counts = {
        "pending": 0,
        "processing": 0,
        "completed": 0,
        "failed": 0,
        "timeout": 0,
        "total": len(all_jobs)
    }

    for _, job_data_json in all_jobs.items():
        try:
            job_data = json.loads(job_data_json)
            status = job_data.get("status", "unknown")

            if status in status_counts:
                status_counts[status] += 1
        except:
            pass

    return status_counts


def main():
    """Main function to parse arguments and execute commands."""
    parser = argparse.ArgumentParser(description="Task Distributor for RAG System")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Status command
    status_parser = subparsers.add_parser("status", help="Display system status")

    # Clear queues command
    clear_parser = subparsers.add_parser("clear", help="Clear message queues")
    clear_parser.add_argument("--queue", "-q", help="Specific queue to clear (or all if not specified)")

    # Reset priority system command
    reset_parser = subparsers.add_parser("reset", help="Reset the priority queue system")

    # Submit test task command
    submit_parser = subparsers.add_parser("submit", help="Submit a test task")
    submit_parser.add_argument("task_type", choices=["inference", "embedding", "whisper", "pdf"],
                               help="Type of task to submit")
    submit_parser.add_argument("--count", "-c", type=int, default=1,
                               help="Number of test tasks to submit")

    # Parse args
    args = parser.parse_args()

    # Execute command
    if args.command == "status":
        display_system_status()
    elif args.command == "clear":
        if args.queue:
            clear_queues([args.queue])
        else:
            clear_queues()
    elif args.command == "reset":
        reset_priority_system()
    elif args.command == "submit":
        submit_test_task(args.task_type, args.count)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()