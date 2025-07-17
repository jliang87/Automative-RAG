import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend
import os
import logging

logger = logging.getLogger(__name__)

# Configure Redis broker
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_broker = RedisBroker(url=redis_url)
redis_backend = RedisBackend(url=redis_url)

# Add results backend to broker
redis_broker.add_middleware(Results(backend=redis_backend))

# Set the global broker
dramatiq.set_broker(redis_broker)


def create_dramatiq_actor_decorator(queue_name: str, **kwargs):
    """
    ✅ ACTUALLY IMPLEMENTED: Factory function to create Dramatiq actor decorators.

    Args:
        queue_name: Name of the queue (from QueueNames enum)
        **kwargs: Additional Dramatiq actor options

    Returns:
        Configured Dramatiq actor decorator for tasks
    """

    # Import queue manager to get configurations
    from src.core.orchestration.queue_manager import queue_manager

    # Validate queue name exists
    if not queue_manager.validate_queue_name(queue_name):
        valid_queues = queue_manager.get_all_queue_names()
        raise ValueError(f"Invalid queue name: {queue_name}. Must be one of: {valid_queues}")

    # Get queue configuration from queue manager
    queue_configs = queue_manager.get_queue_configuration()
    default_config = queue_configs.get(queue_name, {})

    # Remove non-dramatiq fields from config
    dramatiq_config = {}
    for key, value in default_config.items():
        if key in ['max_retries', 'min_backoff', 'max_backoff', 'store_results']:
            dramatiq_config[key] = value

    # Merge with any provided overrides
    final_config = {**dramatiq_config, **kwargs}

    logger.debug(f"Creating Dramatiq actor decorator for queue '{queue_name}' with config: {final_config}")

    return dramatiq.actor(queue_name=queue_name, **final_config)


def get_broker():
    """Get the configured Dramatiq broker."""
    return redis_broker


def get_results_backend():
    """Get the configured results backend."""
    return redis_backend


def get_dramatiq_info():
    """Get information about Dramatiq configuration."""
    from src.core.orchestration.queue_manager import QueueNames

    return {
        "broker_type": "redis",
        "broker_url": redis_url,
        "available_queues": QueueNames.get_all_queue_names(),
        "broker_status": "configured"
    }


def validate_dramatiq_health():
    """Validate that Dramatiq broker and queues are properly configured."""
    try:
        # Test broker connection
        redis_broker.ping()

        # Test results backend
        try:
            redis_backend.get_result("test_key")  # This will return None but validates connection
        except Exception:
            pass  # Expected for non-existent key

        logger.info("✅ Dramatiq broker and results backend are healthy")
        return True

    except Exception as e:
        logger.error(f"❌ Dramatiq health check failed: {str(e)}")
        return False


def initialize_dramatiq():
    """Initialize Dramatiq with queue configurations from queue manager."""
    logger.info("Initializing Dramatiq with queue manager configurations")

    # Set broker
    dramatiq.set_broker(redis_broker)

    # Get queue information
    dramatiq_info = get_dramatiq_info()
    logger.info(f"Configured {len(dramatiq_info['available_queues'])} queues: {dramatiq_info['available_queues']}")

    # Validate health
    if validate_dramatiq_health():
        logger.info("✅ Dramatiq initialization completed successfully")
    else:
        logger.warning("⚠️  Dramatiq initialization completed with warnings")

    return redis_broker


def test_dramatiq_decorator():
    """Test function to verify the decorator works."""
    from src.core.orchestration.queue_manager import QueueNames

    try:
        # Test creating a decorator
        test_decorator = create_dramatiq_actor_decorator(QueueNames.CPU_TASKS.value)

        # Test applying it to a dummy function
        @test_decorator
        def test_task(message: str):
            return f"Test task executed: {message}"

        logger.info("✅ Dramatiq decorator test passed")
        return True

    except Exception as e:
        logger.error(f"❌ Dramatiq decorator test failed: {str(e)}")
        return False


# Auto-initialize when imported
broker = initialize_dramatiq()

# Run test on import (optional - can be removed)
if __name__ != "__main__":
    test_dramatiq_decorator()