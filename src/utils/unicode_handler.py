import json
import logging
import codecs
import re
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


def decode_unicode_escapes(text: str) -> str:
    """
    Comprehensively decode Unicode escape sequences.
    Handles various forms of Unicode encoding issues common with Chinese text.

    Args:
        text: Text that may contain Unicode escape sequences

    Returns:
        Properly decoded Unicode text
    """
    if not text or not isinstance(text, str):
        return text

    try:
        # Pattern 1: Standard Unicode escapes (\uXXXX)
        if '\\u' in text:
            # Try different decoding approaches

            # Approach 1: JSON decoding (for properly formatted JSON strings)
            if text.startswith('"') and text.endswith('"'):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass

            # Approach 2: Direct Unicode escape decoding
            try:
                return text.encode('utf-8').decode('unicode_escape')
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass

            # Approach 3: Latin1 to Unicode escape decoding
            try:
                return text.encode('latin1').decode('unicode_escape')
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass

            # Approach 4: Codecs Unicode escape decoding
            try:
                return codecs.decode(text, 'unicode_escape')
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass

        # Pattern 2: Double-escaped Unicode (\\\uXXXX)
        if '\\\\u' in text:
            # Fix double escaping first
            text = text.replace('\\\\u', '\\u')
            return decode_unicode_escapes(text)  # Recursive call

        # Pattern 3: URL-encoded Unicode (%uXXXX)
        if '%u' in text:
            def replace_percent_unicode(match):
                code = match.group(1)
                return chr(int(code, 16))

            text = re.sub(r'%u([0-9A-Fa-f]{4})', replace_percent_unicode, text)

        # Pattern 4: Check for encoding issues with Chinese characters
        if any(ord(char) > 127 for char in text):
            # If text contains high Unicode but looks wrong, try re-encoding
            try:
                # Test if current encoding is correct
                text.encode('utf-8')
                return text  # Already proper UTF-8
            except UnicodeEncodeError:
                # Try to fix common encoding issues
                try:
                    return text.encode('latin1').decode('utf-8')
                except (UnicodeDecodeError, UnicodeEncodeError):
                    pass

        return text

    except Exception as e:
        logger.warning(f"Unicode decoding failed for text: {text[:50]}..., error: {e}")
        return text  # Return original if all attempts fail


def decode_unicode_in_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively decode Unicode escape sequences in dictionary values.

    Args:
        data: Dictionary that may contain Unicode escape sequences

    Returns:
        Dictionary with decoded Unicode strings
    """
    if not isinstance(data, dict):
        return data

    decoded_data = {}
    for key, value in data.items():
        if isinstance(value, str):
            decoded_data[key] = decode_unicode_escapes(value)
        elif isinstance(value, dict):
            decoded_data[key] = decode_unicode_in_dict(value)
        elif isinstance(value, list):
            decoded_data[key] = decode_unicode_in_list(value)
        else:
            decoded_data[key] = value

    return decoded_data


def decode_unicode_in_list(data: List[Any]) -> List[Any]:
    """
    Recursively decode Unicode escape sequences in list items.

    Args:
        data: List that may contain Unicode escape sequences

    Returns:
        List with decoded Unicode strings
    """
    if not isinstance(data, list):
        return data

    decoded_data = []
    for item in data:
        if isinstance(item, str):
            decoded_data.append(decode_unicode_escapes(item))
        elif isinstance(item, dict):
            decoded_data.append(decode_unicode_in_dict(item))
        elif isinstance(item, list):
            decoded_data.append(decode_unicode_in_list(item))
        else:
            decoded_data.append(item)

    return decoded_data


def decode_unicode_in_json_result(result: Union[str, Dict, List]) -> Union[str, Dict, List]:
    """
    Decode Unicode escape sequences in job tracker results.

    Args:
        result: Result data that may be JSON string or already parsed

    Returns:
        Result with decoded Unicode
    """
    # If it's a JSON string, parse it first
    if isinstance(result, str):
        try:
            parsed_result = json.loads(result)
            decoded_result = decode_unicode_in_dict(parsed_result) if isinstance(parsed_result, dict) else parsed_result
            return json.dumps(decoded_result, ensure_ascii=False)  # Use ensure_ascii=False for proper Unicode
        except json.JSONDecodeError:
            # If it's not JSON, just decode the string
            return decode_unicode_escapes(result)

    # If it's already a dict or list, decode recursively
    if isinstance(result, dict):
        return decode_unicode_in_dict(result)
    elif isinstance(result, list):
        return decode_unicode_in_list(result)

    return result


def clean_form_input(text: str) -> str:
    """
    Clean text from web forms that might contain Unicode escapes.
    Common when users copy-paste from websites.

    Args:
        text: Text input from web forms

    Returns:
        Cleaned text with proper Unicode
    """
    if not text:
        return text

    # Common issues from copy-paste from websites
    if "\\u" in text:
        return decode_unicode_escapes(text)

    return text


def validate_unicode_cleaning(text: str, field_name: str = "text") -> bool:
    """
    Validate that Unicode cleaning was successful.

    Args:
        text: Text to validate
        field_name: Name of the field for logging

    Returns:
        True if text is clean, False if Unicode escapes remain
    """
    if not isinstance(text, str):
        return True

    if "\\u" in text:
        logger.warning(f"Unicode escapes still present in {field_name}: {text[:100]}...")
        return False

    return True


def test_unicode_decoding():
    """Test function to validate Unicode decoding works correctly"""

    # Test cases with common Unicode escape patterns
    test_cases = [
        # Chinese characters with Unicode escapes
        ("\\u6b3e", "款"),
        ("\\u8f66\\u5728\\u7ebf", "车在线"),
        ("25\\u6b3e\\u8fdc\\u9014\\u88c5\\u9970", "25款远途装饰"),

        # Already properly encoded text
        ("正常中文", "正常中文"),
        ("Normal English", "Normal English"),

        # Mixed content
        ("2023\\u5e74\\u5b9d\\u9a6cX5", "2023年宝马X5"),

        # Double-escaped
        ("\\\\u6b3e", "款"),

        # Complex automotive terms
        ("\\u5b9d\\u9a6cX5\\u53c2\\u6570\\u8868", "宝马X5参数表"),
        ("\\u5965\\u8feaA4\\u8fd0\\u52a8\\u578b", "奥迪A4运动型"),
    ]

    print("Testing Unicode decoding...")
    all_passed = True

    for input_text, expected in test_cases:
        result = decode_unicode_escapes(input_text)
        passed = result == expected
        status = "✓" if passed else "✗"
        print(f"{status} '{input_text}' -> '{result}' (expected: '{expected}')")

        if not passed:
            all_passed = False

    # Test dictionary decoding
    print("\nTesting dictionary decoding...")
    test_dict = {
        "title": "25\\u6b3e\\u8fdc\\u9014\\u88c5\\u9970",
        "author": "\\u8f66\\u5728\\u7ebf",
        "manufacturer": "\\u5b9d\\u9a6c",
        "normal_field": "normal text"
    }

    decoded_dict = decode_unicode_in_dict(test_dict)
    expected_dict = {
        "title": "25款远途装饰",
        "author": "车在线",
        "manufacturer": "宝马",
        "normal_field": "normal text"
    }

    dict_passed = decoded_dict == expected_dict
    status = "✓" if dict_passed else "✗"
    print(f"{status} Dictionary decoding: {dict_passed}")

    if not dict_passed:
        print(f"  Expected: {expected_dict}")
        print(f"  Got: {decoded_dict}")
        all_passed = False

    print(f"\nUnicode decoding test {'PASSED' if all_passed else 'FAILED'}.")
    return all_passed


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_unicode_decoding()