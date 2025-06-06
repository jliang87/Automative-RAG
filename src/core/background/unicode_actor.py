# src/core/background/unicode_actor.py

"""
Dramatiq Unicode handling monkey patch.

This module provides a global fix for Unicode escape sequences in Dramatiq actor parameters.
It patches Dramatiq's actor execution to automatically clean Unicode escapes from all
incoming parameters, ensuring all task functions receive proper UTF-8 strings.

This is the SINGLE POINT where Unicode cleaning happens in the entire system.
"""

import logging
import functools
from typing import Any, Callable, Tuple, Dict
from src.utils.unicode_handler import clean_unicode_escapes

logger = logging.getLogger(__name__)

# Global flag to prevent double-patching
_UNICODE_PATCH_APPLIED = False


def patch_dramatiq_unicode_handling():
    """
    Apply global monkey patch to Dramatiq for automatic Unicode escape cleaning.

    This patches the Dramatiq Actor.__call__ method to automatically clean
    Unicode escapes from all actor parameters before execution.

    This should be called ONCE during application initialization.
    """
    global _UNICODE_PATCH_APPLIED

    if _UNICODE_PATCH_APPLIED:
        logger.info("Dramatiq Unicode patch already applied, skipping")
        return

    try:
        from dramatiq.actor import Actor

        # Store the original actor call method
        original_call = Actor.__call__

        def unicode_cleaned_call(self, *args, **kwargs):
            """
            Enhanced actor call that automatically cleans Unicode escapes.

            This method replaces Actor.__call__ to intercept all actor invocations
            and clean Unicode escape sequences from parameters before execution.
            """
            try:
                # Clean all arguments before passing to the actor
                cleaned_args = clean_unicode_escapes(args)
                cleaned_kwargs = clean_unicode_escapes(kwargs)

                logger.debug(f"Auto-cleaning Unicode escapes for actor: {self.actor_name}")

                # Log if any cleaning was performed (for debugging)
                if args != cleaned_args or kwargs != cleaned_kwargs:
                    logger.debug(f"Unicode cleaning applied to {self.actor_name} parameters")

                # Call the original actor with cleaned arguments
                return original_call(self, *cleaned_args, **cleaned_kwargs)

            except Exception as e:
                logger.error(f"Error in Unicode cleaning for actor {self.actor_name}: {e}")
                # Fall back to original call if cleaning fails
                return original_call(self, *args, **kwargs)

        # Replace the actor call method globally
        Actor.__call__ = unicode_cleaned_call

        _UNICODE_PATCH_APPLIED = True
        logger.info("✅ Dramatiq Unicode cleaning patch applied successfully")
        logger.info("All Dramatiq actors will now automatically receive clean UTF-8 parameters")

    except ImportError as e:
        logger.error(f"Failed to import Dramatiq components: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to apply Dramatiq Unicode patch: {e}")
        raise


def patch_dramatiq_message_decoding():
    """
    Alternative approach: Patch at the message decoding level.

    This patches Dramatiq's Message.decode method to clean Unicode escapes
    when messages are decoded from Redis.

    This is a more fundamental approach but requires careful handling.
    """
    global _UNICODE_PATCH_APPLIED

    if _UNICODE_PATCH_APPLIED:
        logger.info("Dramatiq Unicode patch already applied, skipping message decode patch")
        return

    try:
        from dramatiq.message import Message

        # Store the original decode method
        original_decode = Message.decode

        @classmethod
        def unicode_cleaned_decode(cls, data):
            """
            Enhanced message decode that cleans Unicode escapes from message content.
            """
            try:
                # Call original decode first
                message = original_decode(data)

                # Clean Unicode escapes from message args and kwargs
                # Handle dataclass/immutable objects properly
                if hasattr(message, 'args') and message.args:
                    cleaned_args = clean_unicode_escapes(message.args)
                    # Create new message with cleaned args if needed
                    if cleaned_args != message.args:
                        # Use object.__setattr__ for dataclass/frozen objects
                        object.__setattr__(message, 'args', cleaned_args)

                if hasattr(message, 'kwargs') and message.kwargs:
                    cleaned_kwargs = clean_unicode_escapes(message.kwargs)
                    # Create new message with cleaned kwargs if needed
                    if cleaned_kwargs != message.kwargs:
                        # Use object.__setattr__ for dataclass/frozen objects
                        object.__setattr__(message, 'kwargs', cleaned_kwargs)

                logger.debug(
                    f"Unicode cleaning applied to message for actor: {getattr(message, 'actor_name', 'unknown')}")

                return message

            except Exception as e:
                logger.error(f"Error in Unicode cleaning during message decode: {e}")
                # Fall back to original decode if cleaning fails
                return original_decode(data)

        # Replace the message decode method
        Message.decode = unicode_cleaned_decode

        _UNICODE_PATCH_APPLIED = True
        logger.info("✅ Dramatiq message decode Unicode patch applied successfully")

    except ImportError as e:
        logger.error(f"Failed to import Dramatiq Message: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to apply Dramatiq message decode patch: {e}")
        raise


def patch_dramatiq_consumer():
    """
    Alternative approach: Patch at the consumer level.

    This patches the Dramatiq consumer to clean Unicode escapes when
    processing messages from the queue.
    """
    global _UNICODE_PATCH_APPLIED

    if _UNICODE_PATCH_APPLIED:
        logger.info("Dramatiq Unicode patch already applied, skipping consumer patch")
        return

    try:
        from dramatiq.worker import Worker

        # Store the original process_message method
        original_process_message = Worker.process_message

        def unicode_cleaned_process_message(self, message):
            """
            Enhanced message processing that cleans Unicode escapes.
            """
            try:
                # Clean Unicode escapes from message before processing
                if hasattr(message, 'args') and message.args:
                    message.args = clean_unicode_escapes(message.args)

                if hasattr(message, 'kwargs') and message.kwargs:
                    message.kwargs = clean_unicode_escapes(message.kwargs)

                logger.debug(
                    f"Unicode cleaning applied during message processing for: {getattr(message, 'actor_name', 'unknown')}")

                # Call original process_message with cleaned message
                return original_process_message(self, message)

            except Exception as e:
                logger.error(f"Error in Unicode cleaning during message processing: {e}")
                # Fall back to original processing if cleaning fails
                return original_process_message(self, message)

        # Replace the process_message method
        Worker.process_message = unicode_cleaned_process_message

        _UNICODE_PATCH_APPLIED = True
        logger.info("✅ Dramatiq consumer Unicode patch applied successfully")

    except ImportError as e:
        logger.error(f"Failed to import Dramatiq Worker: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to apply Dramatiq consumer patch: {e}")
        raise


def test_unicode_patch():
    """
    Test function to verify the Unicode patch is working correctly.

    This creates a test actor and verifies that Unicode escapes are
    automatically cleaned from parameters.
    """
    if not _UNICODE_PATCH_APPLIED:
        logger.warning("Unicode patch not applied, test may fail")

    try:
        import dramatiq

        # Create a test actor
        @dramatiq.actor(queue_name="test_unicode_queue")
        def test_unicode_actor(test_data: dict, test_string: str):
            """Test actor to verify Unicode cleaning."""
            logger.info(f"Test actor received data: {repr(test_data)}")
            logger.info(f"Test actor received string: {repr(test_string)}")

            # Check if the data contains proper Chinese characters (not escapes)
            title = test_data.get("title", "")
            if "万" in title and "\\u" not in title and "\\x" not in title:
                logger.info("✅ Unicode cleaning working - received proper Chinese characters")
                return {"success": True, "title": title}
            else:
                logger.error("❌ Unicode cleaning failed - still contains escapes or missing characters")
                return {"success": False, "title": title}

        # Test data with Unicode escapes (simulating what Dramatiq would receive)
        test_data = {
            "title": "15.2\\u4e07\\u540e\\u6094\\u63d0\\u4e86\\u661f\\u8d8aL",  # Should become "15.2万后悔提了星越L"
            "author": "\\u5c0f\\u80e1\\u5b50\\u8bf4\\u8f66"  # Should become "小胡子说车"
        }
        test_string = "\\u6d4b\\u8bd5\\u5b57\\u7b26\\u4e32"  # Should become "测试字符串"

        logger.info("Testing Unicode patch with escaped data...")
        logger.info(f"Input data: {repr(test_data)}")
        logger.info(f"Input string: {repr(test_string)}")

        # This would normally be called by Dramatiq worker, but we can test the wrapper
        # Note: In real usage, this would be dispatched through Redis
        result = test_unicode_actor(test_data, test_string)

        logger.info(f"Test result: {result}")

        if result.get("success"):
            logger.info("🎉 Unicode patch test PASSED!")
            return True
        else:
            logger.error("❌ Unicode patch test FAILED!")
            return False

    except Exception as e:
        logger.error(f"Unicode patch test failed with error: {e}")
        return False


def verify_patch_status():
    """
    Verify that the Unicode patch has been applied correctly.

    Returns:
        dict: Status information about the patch
    """
    status = {
        "patch_applied": _UNICODE_PATCH_APPLIED,
        "timestamp": None,
        "actor_patched": False,
        "message_patched": False,
        "consumer_patched": False
    }

    try:
        from dramatiq.actor import Actor
        from dramatiq.message import Message
        from dramatiq.worker import Worker

        # Check if Actor.__call__ has been patched
        if hasattr(Actor.__call__, '__name__'):
            status["actor_patched"] = Actor.__call__.__name__ == "unicode_cleaned_call"

        # Check if Message.decode has been patched
        if hasattr(Message.decode, '__name__'):
            status["message_patched"] = Message.decode.__name__ == "unicode_cleaned_decode"

        # Check if Worker.process_message has been patched
        if hasattr(Worker.process_message, '__name__'):
            status["consumer_patched"] = Worker.process_message.__name__ == "unicode_cleaned_process_message"

        logger.info(f"Patch status: {status}")

    except Exception as e:
        logger.error(f"Error checking patch status: {e}")
        status["error"] = str(e)

    return status


def get_unicode_cleaning_stats():
    """
    Get statistics about Unicode cleaning operations.

    This could be extended to track how often cleaning is performed,
    what types of escapes are found, etc.
    """
    # This is a placeholder for potential monitoring/metrics
    return {
        "patch_applied": _UNICODE_PATCH_APPLIED,
        "version": "1.0.0",
        "supported_escape_types": ["\\uXXXX", "\\xXX"],
        "description": "Global Dramatiq Unicode escape cleaning"
    }


# Main initialization function
def initialize_unicode_handling():
    """
    Main function to initialize Unicode handling for Dramatiq.

    This should be called once during application startup, preferably
    in common.py or __init__.py of the background package.
    """
    logger.info("Initializing Dramatiq Unicode handling...")

    try:
        # Apply the main patch (choose one approach)
        patch_dramatiq_unicode_handling()  # Recommended approach

        # Verify the patch was applied
        status = verify_patch_status()
        if status["patch_applied"]:
            logger.info("✅ Dramatiq Unicode handling initialized successfully")

            # Run a quick test if in development mode
            import os
            if os.environ.get("ENVIRONMENT") == "development":
                logger.info("Running Unicode patch test in development mode...")
                test_unicode_patch()
        else:
            logger.error("❌ Failed to initialize Dramatiq Unicode handling")

    except Exception as e:
        logger.error(f"Failed to initialize Unicode handling: {e}")
        raise


if __name__ == "__main__":
    # If run directly, perform initialization and testing
    print("=== Dramatiq Unicode Handling Test ===")

    # Initialize
    initialize_unicode_handling()

    # Test
    test_result = test_unicode_patch()

    # Show status
    status = verify_patch_status()
    print(f"Patch Status: {status}")

    # Show stats
    stats = get_unicode_cleaning_stats()
    print(f"Stats: {stats}")

    if test_result:
        print("🎉 All tests passed!")
    else:
        print("❌ Tests failed!")