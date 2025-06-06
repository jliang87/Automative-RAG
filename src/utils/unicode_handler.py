import logging
import codecs
import json
from typing import Any, Dict

logger = logging.getLogger(__name__)


def decode_unicode_escapes(text: str) -> str:
    """
    Decode Unicode escape sequences in text.

    This is the CORE function used by the global Dramatiq patch.
    Must remain exactly as is.
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


def clean_unicode_escapes(data: Any) -> Any:
    """
    Recursively clean Unicode escapes from any data structure.

    This is used by the global Dramatiq patch to clean ALL actor parameters.
    """
    if isinstance(data, dict):
        return {key: clean_unicode_escapes(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_unicode_escapes(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(clean_unicode_escapes(item) for item in data)
    elif isinstance(data, str):
        return decode_unicode_escapes(data)
    else:
        return data


def safe_json_dumps(data: Any, **kwargs) -> str:
    """
    JSON dumping with proper Unicode handling.
    Always preserves Chinese characters.
    """
    kwargs['ensure_ascii'] = False
    try:
        return json.dumps(data, **kwargs)
    except (TypeError, ValueError) as e:
        logger.error(f"JSON serialization failed: {e}")
        cleaned_data = clean_unicode_escapes(data)
        return json.dumps(cleaned_data, **kwargs)
