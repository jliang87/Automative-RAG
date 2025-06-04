import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from datetime import datetime


__all__ = [
    "generate_unique_id",
    "clean_text",
    "extract_year_from_text",
    "extract_metadata_from_text",
    "format_time",
    "parse_youtube_url",
    "parse_bilibili_url",
    "chunk_text",
    "get_file_extension",
    "is_valid_file_type",
    "safe_json_loads",
    "retry_with_backoff"
]


def generate_unique_id(prefix: str = "") -> str:
    """
    Generate a unique ID with optional prefix.

    Args:
        prefix: Optional prefix string

    Returns:
        Unique ID string
    """
    unique_id = str(uuid.uuid4())
    if prefix:
        return f"{prefix}-{unique_id}"
    return unique_id


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace, newlines, etc.
    ENHANCED with Unicode handling.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Apply Unicode decoding if needed
    if isinstance(text, str) and "\\u" in text:
        try:
            from src.utils.unicode_handler import decode_unicode_escapes
            text = decode_unicode_escapes(text)
        except ImportError:
            pass  # Unicode handler not available

    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)

    # Strip whitespace from beginning and end
    text = text.strip()

    # Replace common Unicode characters
    replacements = {
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u00a0': ' ',  # Non-breaking space
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def extract_year_from_text(text: str) -> Optional[int]:
    """
    Extract a year (between 1900 and 2100) from text.
    ENHANCED with Unicode handling.

    Args:
        text: Text to search for year

    Returns:
        Year as integer or None if not found
    """
    # Apply Unicode decoding if needed
    if isinstance(text, str) and "\\u" in text:
        try:
            from src.utils.unicode_handler import decode_unicode_escapes
            text = decode_unicode_escapes(text)
        except ImportError:
            pass

    year_match = re.search(r'(19\d{2}|20\d{2})', text)
    if year_match:
        return int(year_match.group(0))
    return None


def extract_metadata_from_text(text: str) -> Dict[str, Any]:
    """
    Extract automotive metadata from text with comprehensive Unicode handling.
    ENHANCED to handle both Chinese and English automotive terms.

    Args:
        text: Text to extract metadata from

    Returns:
        Dictionary of extracted metadata
    """
    # Apply Unicode decoding if needed
    if isinstance(text, str) and "\\u" in text:
        try:
            from src.utils.unicode_handler import decode_unicode_escapes
            text = decode_unicode_escapes(text)
        except ImportError:
            pass

    metadata = {}
    text_lower = text.lower()

    # Extract year
    year = extract_year_from_text(text)
    if year:
        metadata["year"] = year

    # ENHANCED: Chinese and English manufacturer detection
    manufacturers = [
        # Chinese names with English alternatives
        ("宝马", "BMW"), ("奔驰", "Mercedes-Benz"), ("奥迪", "Audi"),
        ("丰田", "Toyota"), ("本田", "Honda"), ("大众", "Volkswagen"),
        ("福特", "Ford"), ("雪佛兰", "Chevrolet"), ("日产", "Nissan"),
        ("现代", "Hyundai"), ("起亚", "Kia"), ("斯巴鲁", "Subaru"),
        ("马自达", "Mazda"), ("特斯拉", "Tesla"), ("沃尔沃", "Volvo"),
        ("捷豹", "Jaguar"), ("路虎", "Land Rover"), ("雷克萨斯", "Lexus"),
        ("讴歌", "Acura"), ("英菲尼迪", "Infiniti"), ("凯迪拉克", "Cadillac"),
        ("吉普", "Jeep"), ("法拉利", "Ferrari"), ("兰博基尼", "Lamborghini"),
        ("保时捷", "Porsche"), ("玛莎拉蒂", "Maserati"), ("阿斯顿马丁", "Aston Martin"),

        # Additional English manufacturers
        ("Lincoln", "Lincoln"), ("Buick", "Buick"), ("GMC", "GMC"),
        ("Dodge", "Dodge"), ("Chrysler", "Chrysler"), ("Mitsubishi", "Mitsubishi"),
        ("Suzuki", "Suzuki"), ("Isuzu", "Isuzu"), ("Alfa Romeo", "Alfa Romeo"),
        ("Bentley", "Bentley"), ("Rolls-Royce", "Rolls-Royce"), ("McLaren", "McLaren")
    ]

    # Look for manufacturers (prioritize Chinese names)
    for chinese_name, english_name in manufacturers:
        # Check for Chinese name first
        if chinese_name in text:
            metadata["manufacturer"] = chinese_name  # Prefer Chinese name
            break
        # Check for English name (case insensitive)
        elif english_name.lower() in text_lower:
            metadata["manufacturer"] = english_name
            break

    # ENHANCED: Chinese category detection
    categories = {
        "轿车": ["轿车", "sedan", "saloon"],
        "SUV": ["suv", "越野车", "运动型多用途车", "sport utility vehicle", "crossover"],
        "卡车": ["truck", "pickup", "卡车", "皮卡", "货车"],
        "跑车": ["sports car", "supercar", "hypercar", "跑车", "运动车"],
        "面包车": ["minivan", "van", "面包车", "mpv", "多功能车"],
        "轿跑": ["coupe", "coupé", "双门轿跑", "轿跑车"],
        "敞篷车": ["convertible", "cabriolet", "敞篷车", "软顶车"],
        "掀背车": ["hatchback", "hot hatch", "掀背车", "两厢车"],
        "旅行车": ["wagon", "estate", "旅行车", "瓦罐车"],
    }

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in text_lower:
                metadata["category"] = category
                break
        if "category" in metadata:
            break

    # ENHANCED: Chinese engine type detection
    engine_types = {
        "汽油": ["汽油", "gasoline", "petrol", "gas engine", "汽油机"],
        "柴油": ["柴油", "diesel", "柴油机"],
        "电动": ["电动", "electric", "ev", "纯电", "电池", "battery", "pure electric"],
        "混合动力": ["混合动力", "hybrid", "油电混合", "插电混合", "phev", "plug-in hybrid"],
        "氢燃料": ["氢燃料", "hydrogen", "fuel cell", "氢气", "燃料电池"],
    }

    for engine_type, keywords in engine_types.items():
        for keyword in keywords:
            if keyword in text_lower:
                metadata["engine_type"] = engine_type
                break
        if "engine_type" in metadata:
            break

    # ENHANCED: Chinese transmission type detection
    transmission_types = {
        "自动": ["自动", "automatic", "auto", "自动挡", "自动变速箱"],
        "手动": ["手动", "manual", "stick", "手动挡", "手动变速箱", "manual transmission"],
        "CVT": ["cvt", "无级变速", "continuously variable", "无级变速箱"],
        "双离合": ["dct", "dual-clutch", "双离合", "双离合变速箱"],
    }

    for transmission, keywords in transmission_types.items():
        for keyword in keywords:
            if keyword in text_lower:
                metadata["transmission"] = transmission
                break
        if "transmission" in metadata:
            break

    # ENHANCED: Extract Chinese model names (common patterns)
    # This is more complex for Chinese models, so we use pattern matching
    if "manufacturer" in metadata:
        manufacturer = metadata["manufacturer"]

        # Common Chinese model patterns
        model_patterns = {
            "宝马": [r"(X[1-7]|[1-8]系|i[3-8]|Z4)", r"(X[1-7]|Series [1-8]|i[3-8]|Z4)"],
            "奔驰": [r"([A-Z]级|[A-Z]-Class|GLA|GLC|GLE|GLS|AMG)", r"([A-Z]级|[A-Z]-Class|GLA|GLC|GLE|GLS|AMG)"],
            "奥迪": [r"(A[1-8]|Q[2-8]|TT|R8)", r"(A[1-8]|Q[2-8]|TT|R8)"],
            "丰田": [r"(凯美瑞|卡罗拉|汉兰达|普拉多|陆地巡洋舰)", r"(Camry|Corolla|Highlander|Prado|Land Cruiser)"],
            "本田": [r"(雅阁|思域|CR-V|奥德赛)", r"(Accord|Civic|CR-V|Odyssey)"],
            # Add more patterns as needed
        }

        if manufacturer in model_patterns:
            for pattern in model_patterns[manufacturer]:
                model_match = re.search(pattern, text, re.IGNORECASE)
                if model_match:
                    metadata["model"] = model_match.group(1)
                    break

    return metadata


def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def parse_youtube_url(url: str) -> Optional[str]:
    """
    Extract video ID from a YouTube URL.

    Args:
        url: YouTube URL

    Returns:
        Video ID or None if invalid
    """
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def parse_bilibili_url(url: str) -> Optional[str]:
    """
    Extract video ID from a Bilibili URL.

    Args:
        url: Bilibili URL

    Returns:
        Video ID or None if invalid
    """
    # BV ID pattern
    bv_pattern = r'(?:bilibili\.com/video/)(BV[a-zA-Z0-9]+)'
    match = re.search(bv_pattern, url)
    if match:
        return match.group(1)

    # AV ID pattern
    av_pattern = r'(?:bilibili\.com/video/av|bilibili\.com/av)(\d+)'
    match = re.search(av_pattern, url)
    if match:
        return f"av{match.group(1)}"

    return None


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into chunks with overlap.
    ENHANCED with Unicode handling.

    Args:
        text: Text to split
        chunk_size: Size of each chunk
        overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    # Clean the text first (includes Unicode handling)
    text = clean_text(text)

    chunks = []
    start = 0
    text_length = len(text)

    # Try to split on paragraph breaks first
    paragraphs = text.split('\n\n')

    # If we have a reasonable number of paragraphs, use them as a starting point
    if len(paragraphs) > 1 and all(len(p) < chunk_size * 2 for p in paragraphs):
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph exceeds chunk size, save current chunk and start new one
            if current_size + len(paragraph) > chunk_size and current_size > 0:
                chunks.append(' '.join(current_chunk))

                # Handle overlap by keeping some paragraphs
                overlap_size = 0
                overlap_chunks = []

                # Work backwards to include paragraphs for overlap
                for i in range(len(current_chunk) - 1, -1, -1):
                    overlap_size += len(current_chunk[i])
                    overlap_chunks.insert(0, current_chunk[i])
                    if overlap_size >= overlap:
                        break

                current_chunk = overlap_chunks
                current_size = sum(len(p) for p in current_chunk)

            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_size += len(paragraph)

        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    else:
        # Fall back to character-based chunking
        while start < text_length:
            end = min(start + chunk_size, text_length)

            # Try to break at a sentence end
            if end < text_length:
                # Look for sentence end within the last 100 characters of the chunk
                search_start = max(end - 100, start)
                search_text = text[search_start:end]

                # Find the last sentence end (enhanced for Chinese text)
                sentence_ends = list(re.finditer(r'[.!?。！？]\s*', search_text))
                if sentence_ends:
                    # Adjust the end to break at the sentence end
                    last_end = sentence_ends[-1].end()
                    end = search_start + last_end

            # Add the chunk
            chunks.append(text[start:end])

            # Move start position for next chunk, considering overlap
            start = end - overlap

    return chunks


def get_file_extension(filename: str) -> str:
    """
    Get the file extension from a filename.

    Args:
        filename: Name of the file

    Returns:
        File extension (lowercase)
    """
    _, ext = os.path.splitext(filename)
    return ext.lower().lstrip(".")


def is_valid_file_type(filename: str, allowed_types: List[str]) -> bool:
    """
    Check if a file has an allowed extension.

    Args:
        filename: Name of the file
        allowed_types: List of allowed extensions

    Returns:
        True if file type is allowed
    """
    ext = get_file_extension(filename)
    return ext in [t.lstrip(".").lower() for t in allowed_types]


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely load JSON string with Unicode handling, returning default value on error.

    Args:
        json_str: JSON string to parse
        default: Default value to return on error

    Returns:
        Parsed JSON object or default value
    """
    try:
        # Apply Unicode decoding if needed
        if isinstance(json_str, str) and "\\u" in json_str:
            try:
                from src.utils.unicode_handler import decode_unicode_escapes
                json_str = decode_unicode_escapes(json_str)
            except ImportError:
                pass

        return json.loads(json_str)
    except (ValueError, TypeError):
        return default


def retry_with_backoff(
    func: callable,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Any:
    """
    Execute a function with exponential backoff retry logic.

    Args:
        func: Function to execute
        max_retries: Maximum number of retries
        initial_backoff: Initial backoff in seconds
        backoff_factor: Factor to multiply backoff on each retry
        exceptions: Tuple of exceptions to catch

    Returns:
        Result of the function call
    """
    retries = 0
    backoff = initial_backoff

    while True:
        try:
            return func()
        except exceptions as e:
            retries += 1
            if retries >= max_retries:
                raise

            # Wait with exponential backoff
            time.sleep(backoff)
            backoff *= backoff_factor