# Enhanced src/utils/unicode_handler.py
# Consolidating all Unicode handling functionality

"""
Comprehensive Unicode handling utilities for proper Chinese character support.
Handles Unicode escapes, encoding issues, and validation.
"""

import logging
import codecs
import json
import re
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


# ===================================================================
# EXISTING FUNCTIONS (keep these as they're already used in your code)
# ===================================================================

def decode_unicode_escapes(text: str) -> str:
    """
    Decode Unicode escape sequences in text.
    Handles both \\uXXXX and \\xXX formats.

    This function is already used throughout your codebase.
    """
    if not isinstance(text, str):
        return text

    if "\\u" not in text and "\\x" not in text:
        return text

    try:
        # Handle \\uXXXX sequences
        if "\\u" in text:
            try:
                return text.encode().decode('unicode_escape')
            except (UnicodeDecodeError, UnicodeError):
                pass

        # Handle \\xXX sequences  
        if "\\x" in text:
            try:
                return text.encode('latin1').decode('utf-8')
            except (UnicodeDecodeError, UnicodeError):
                pass

        # Fallback using codecs
        return codecs.decode(text, 'unicode_escape')

    except Exception as e:
        logger.warning(f"Failed to decode Unicode escapes in: {repr(text)}, error: {e}")
        return text


def decode_unicode_in_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply Unicode decoding to all string values in a dictionary.

    This function is already used in your codebase.
    """
    if not isinstance(data, dict):
        return data

    decoded_dict = {}
    for key, value in data.items():
        if isinstance(value, str):
            decoded_dict[key] = decode_unicode_escapes(value)
        elif isinstance(value, dict):
            decoded_dict[key] = decode_unicode_in_dict(value)
        elif isinstance(value, list):
            decoded_dict[key] = [decode_unicode_escapes(item) if isinstance(item, str) else item for item in value]
        else:
            decoded_dict[key] = value

    return decoded_dict


def decode_unicode_in_json_result(result: Any) -> Any:
    """
    Apply Unicode decoding to JSON result data.

    This function is already used in your job_tracker.
    """
    if isinstance(result, dict):
        return decode_unicode_in_dict(result)
    elif isinstance(result, list):
        return [decode_unicode_in_json_result(item) for item in result]
    elif isinstance(result, str):
        return decode_unicode_escapes(result)
    else:
        return result


def validate_unicode_cleaning(text: str, field_name: str = "field") -> bool:
    """
    Validate that Unicode decoding was successful.

    This function is already used in your codebase for validation.
    """
    if not isinstance(text, str):
        return True

    # Check for remaining escape sequences
    if "\\u" in text or "\\x" in text:
        logger.warning(f"Unicode escapes still present in {field_name}: {repr(text)}")
        return False

    # Check for empty content
    if not text.strip():
        logger.warning(f"Field {field_name} is empty after Unicode cleaning")
        return False

    return True


# ===================================================================
# ENHANCED FUNCTIONS (new improved versions for better handling)
# ===================================================================

def clean_unicode_escapes(data: Any) -> Any:
    """
    ENHANCED: Recursively clean Unicode escapes from any data structure.

    This is the new comprehensive function that handles the Redis escape issue.
    Replaces the individual functions with a more robust approach.

    Args:
        data: Any data structure (dict, list, str, or primitive)

    Returns:
        Data with Unicode escapes decoded to proper characters

    Examples:
        >>> clean_unicode_escapes("15.2\\u4e07")
        "15.2万"

        >>> clean_unicode_escapes({"title": "\\xe4\\xb8\\x87"})
        {"title": "万"}
    """
    if isinstance(data, dict):
        return {key: clean_unicode_escapes(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_unicode_escapes(item) for item in data]
    elif isinstance(data, str):
        return _robust_decode_string(data)
    else:
        return data


def _robust_decode_string(text: str) -> str:
    """
    ENHANCED: More robust string decoding with multiple strategies.

    This handles the specific escape formats seen in your Redis data.
    """
    if not isinstance(text, str) or ('\\u' not in text and '\\x' not in text):
        return text

    original_text = text

    # Strategy 1: Handle \\xXX byte sequences (like from your Redis data)
    if '\\x' in text:
        try:
            # This handles sequences like \\xe4\\xb8\\x87 -> 万
            decoded = text.encode('latin1').decode('utf-8')
            if decoded != text and not _has_escape_sequences(decoded):
                logger.debug(f"Decoded using latin1->utf8: {repr(text)} -> {repr(decoded)}")
                return decoded
        except (UnicodeDecodeError, UnicodeError):
            pass

    # Strategy 2: Handle \\uXXXX Unicode sequences (like from Dramatiq)
    if '\\u' in text:
        try:
            decoded = text.encode().decode('unicode_escape')
            if decoded != text and not _has_escape_sequences(decoded):
                logger.debug(f"Decoded using unicode_escape: {repr(text)} -> {repr(decoded)}")
                return decoded
        except (UnicodeDecodeError, UnicodeError):
            pass

    # Strategy 3: Use codecs as fallback
    try:
        decoded = codecs.decode(text, 'unicode_escape')
        if isinstance(decoded, bytes):
            decoded = decoded.decode('utf-8')
        if decoded != text and not _has_escape_sequences(decoded):
            logger.debug(f"Decoded using codecs: {repr(text)} -> {repr(decoded)}")
            return decoded
    except (UnicodeDecodeError, UnicodeError, AttributeError):
        pass

    # Strategy 4: Try the existing decode_unicode_escapes function
    try:
        decoded = decode_unicode_escapes(text)
        if decoded != text:
            logger.debug(f"Decoded using existing function: {repr(text)} -> {repr(decoded)}")
            return decoded
    except Exception:
        pass

    # If all strategies failed, log and return original
    if _has_escape_sequences(text):
        logger.warning(f"All decoding strategies failed for: {repr(original_text)}")

    return original_text


def _has_escape_sequences(text: str) -> bool:
    """Check if text still contains escape sequences."""
    return isinstance(text, str) and ('\\u' in text or '\\x' in text)


def clean_video_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    ENHANCED: Specialized function to clean video metadata with detailed logging.

    This is specifically for video processing tasks.

    Args:
        metadata: Video metadata dictionary from yt-dlp

    Returns:
        Cleaned metadata dictionary with proper Chinese characters
    """
    if not isinstance(metadata, dict):
        return metadata

    logger.info(f"Cleaning video metadata with {len(metadata)} fields")

    cleaned = {}

    for key, value in metadata.items():
        if isinstance(value, str) and _has_escape_sequences(value):
            logger.info(f"Cleaning Unicode escapes from {key}: {repr(value)}")
            cleaned_value = clean_unicode_escapes(value)
            logger.info(f"Cleaned {key}: {repr(cleaned_value)}")

            # Validate the cleaning worked
            if validate_unicode_cleaning(cleaned_value, key):
                cleaned[key] = cleaned_value
            else:
                logger.error(f"Unicode cleaning failed for {key}, using original value")
                cleaned[key] = value
        else:
            cleaned[key] = clean_unicode_escapes(value)

    logger.info(f"Video metadata cleaning completed")
    return cleaned


def safe_json_dumps(data: Any, **kwargs) -> str:
    """
    ENHANCED: Safe JSON dumping with proper Unicode handling.

    Always uses ensure_ascii=False to preserve Chinese characters.

    Args:
        data: Data to serialize
        **kwargs: Additional arguments for json.dumps

    Returns:
        JSON string with proper Unicode characters
    """
    # Force ensure_ascii=False to preserve Chinese characters
    kwargs['ensure_ascii'] = False

    try:
        return json.dumps(data, **kwargs)
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization failed: {e}")
        # Try with cleaned data
        cleaned_data = clean_unicode_escapes(data)
        return json.dumps(cleaned_data, **kwargs)


def safe_json_loads(json_str: str) -> Any:
    """
    ENHANCED: Safe JSON loading with Unicode cleaning.

    Automatically cleans any Unicode escapes in the loaded data.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed data with cleaned Unicode characters
    """
    try:
        data = json.loads(json_str)
        return clean_unicode_escapes(data)
    except (TypeError, ValueError) as e:
        logger.error(f"JSON parsing failed: {e}")
        raise


# ===================================================================
# COMPATIBILITY FUNCTIONS (for your existing code)
# ===================================================================

# Keep the old function names for backward compatibility
def decode_unicode_in_dict_new(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward compatibility wrapper.
    Use clean_unicode_escapes() for new code.
    """
    return clean_unicode_escapes(data)


def decode_unicode_in_json_result_new(result: Any) -> Any:
    """
    Backward compatibility wrapper.
    Use clean_unicode_escapes() for new code.
    """
    return clean_unicode_escapes(result)


# ===================================================================
# TESTING AND VALIDATION
# ===================================================================

def test_unicode_handling():
    """
    Comprehensive test suite for all Unicode handling functions.

    Tests both existing and new functionality.
    """

    test_cases = [
        # Test case 1: Your actual Redis data format
        {
            "input": "15.2\\xe4\\xb8\\x87\\xe5\\x90\\x8e\\xe6\\x82\\x94\\xe6\\x8f\\x90\\xe4\\xba\\x86\\xe6\\x98\\x9f\\xe8\\xb6\\x8aL",
            "expected": "15.2万后悔提了星越L",
            "description": "Redis byte escape sequences"
        },

        # Test case 2: Dramatiq Unicode sequences
        {
            "input": "15.2\\u4e07\\u540e\\u6094\\u63d0\\u4e86\\u661f\\u8d8aL",
            "expected": "15.2万后悔提了星越L",
            "description": "Dramatiq Unicode escapes"
        },

        # Test case 3: Author name from Redis
        {
            "input": "\\xe5\\xb0\\x8f\\xe8\\x83\\xa1\\xe5\\xad\\x90\\xe8\\xaf\\xb4\\xe8\\xbd\\xa6",
            "expected": "小胡子说车",
            "description": "Author name byte escapes"
        },

        # Test case 4: Complex nested structure (like job results)
        {
            "input": {
                "video_metadata": {
                    "title": "\\xe6\\xb5\\x8b\\xe8\\xaf\\x95\\xe8\\xa7\\x86\\xe9\\xa2\\x91",
                    "author": "\\u5c0f\\u80e1\\u5b50"
                },
                "transcript": "\\xe8\\xbf\\x99\\xe6\\x98\\xaf\\xe4\\xb8\\x80\\xe4\\xb8\\xaa\\xe6\\xb5\\x8b\\xe8\\xaf\\x95",
                "normal_field": "No escapes here"
            },
            "expected": {
                "video_metadata": {
                    "title": "测试视频",
                    "author": "小胡子"
                },
                "transcript": "这是一个测试",
                "normal_field": "No escapes here"
            },
            "description": "Complex job result structure"
        }
    ]

    print("=== TESTING ENHANCED UNICODE HANDLER ===\n")

    all_passed = True

    # Test the new clean_unicode_escapes function
    print("Testing clean_unicode_escapes():")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Input: {repr(test_case['input'])}")

        result = clean_unicode_escapes(test_case['input'])
        print(f"Output: {repr(result)}")
        print(f"Expected: {repr(test_case['expected'])}")

        if result == test_case['expected']:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
            all_passed = False

    # Test backward compatibility
    print("\n" + "=" * 50)
    print("Testing backward compatibility:")

    test_dict = {"title": "\\xe6\\xb5\\x8b\\xe8\\xaf\\x95"}

    old_result = decode_unicode_in_dict(test_dict)
    new_result = clean_unicode_escapes(test_dict)

    print(f"Old function result: {repr(old_result)}")
    print(f"New function result: {repr(new_result)}")

    if old_result.get('title') == new_result.get('title'):
        print("✅ Backward compatibility maintained")
    else:
        print("❌ Backward compatibility broken")
        all_passed = False

    # Test JSON functions
    print("\n" + "=" * 50)
    print("Testing JSON functions:")

    test_data = {"title": "测试标题", "author": "作者名"}
    json_str = safe_json_dumps(test_data)
    loaded_data = safe_json_loads(json_str)

    print(f"Original: {repr(test_data)}")
    print(f"JSON: {repr(json_str)}")
    print(f"Loaded: {repr(loaded_data)}")

    if loaded_data == test_data:
        print("✅ JSON functions working")
    else:
        print("❌ JSON functions failed")
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Unicode handler is working correctly.")
    else:
        print("❌ Some tests failed! Check the implementation.")

    return all_passed


if __name__ == "__main__":
    test_unicode_handling()

# ===================================================================
# USAGE EXAMPLES FOR YOUR CODEBASE
# ===================================================================

"""
Usage in your existing code:

1. In job_tracker.py:
   from src.utils.unicode_handler import clean_unicode_escapes

   # Replace _clean_unicode_escapes with:
   cleaned_result = clean_unicode_escapes(result)

2. In job_chain.py:
   from src.utils.unicode_handler import clean_unicode_escapes, clean_video_metadata

   # For general data cleaning:
   cleaned_result = clean_unicode_escapes(result)

   # For video metadata specifically:
   cleaned_metadata = clean_video_metadata(video_metadata)

3. For JSON operations:
   from src.utils.unicode_handler import safe_json_dumps, safe_json_loads

   # Instead of json.dumps(data, ensure_ascii=False):
   json_str = safe_json_dumps(data)

   # Instead of json.loads(json_str):
   data = safe_json_loads(json_str)

The existing functions (decode_unicode_escapes, decode_unicode_in_dict, etc.) 
continue to work exactly as before for backward compatibility.
"""