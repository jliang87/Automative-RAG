#!/bin/bash
# GPU worker monitoring and management script
# Place this in the project root directory and make it executable with:
# chmod +x gpu_worker_manager.sh

# Configuration
MAX_RESTART_COUNT=5
RESTART_COOLDOWN=30  # seconds
LOG_FILE="logs/gpu_worker_manager.log"
WORKER_NAME="worker-gpu"

# Create log directory if it doesn't exist
mkdir -p logs

log() {
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1" | tee -a "$LOG_FILE"
}

restart_worker() {
    log "Restarting GPU worker container..."
    docker-compose restart $WORKER_NAME
    if [ $? -eq 0 ]; then
        log "GPU worker restarted successfully"
        return 0
    else
        log "Failed to restart GPU worker"
        return 1
    fi
}

check_worker_health() {
    # Check if worker is running and healthy
    container_status=$(docker-compose ps -q $WORKER_NAME)
    if [ -z "$container_status" ]; then
        log "GPU worker container is not running"
        return 1
    fi

    # Check logs for OOM errors
    if docker-compose logs --tail=50 $WORKER_NAME | grep -q "CUDA out of memory"; then
        log "CUDA out of memory error detected"
        return 1
    fi

    # Check if worker is processing tasks
    if docker-compose logs --tail=20 $WORKER_NAME | grep -q "No tasks were processed in the last"; then
        log "Worker appears to be stalled"
        return 1
    fi

    # Worker seems healthy
    return 0
}

# Main monitoring loop
log "Starting GPU worker monitoring"
restart_count=0

while true; do
    if ! check_worker_health; then
        # Worker is unhealthy, attempt restart
        log "Unhealthy worker detected. Restart count: $restart_count/$MAX_RESTART_COUNT"

        if [ $restart_count -lt $MAX_RESTART_COUNT ]; then
            restart_worker
            restart_count=$((restart_count + 1))
            log "Cooling down for $RESTART_COOLDOWN seconds..."
            sleep $RESTART_COOLDOWN
        else
            log "Maximum restart count reached ($MAX_RESTART_COUNT). Manual intervention required."
            # Optionally send an alert here (email, notification, etc.)

            # Reset restart counter after a longer period
            log "Resetting restart counter after 5 minutes..."
            sleep 300
            restart_count=0
        fi
    else
        # Worker is healthy, reset restart counter if it was previously incremented
        if [ $restart_count -gt 0 ]; then
            log "Worker appears to be healthy again. Resetting restart counter."
            restart_count=0
        fi

        # Regular check interval
        sleep 60
    fi
done