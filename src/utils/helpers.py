"""
Helper utility functions for the Automotive Specs RAG system.
"""

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
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
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
    
    Args:
        text: Text to search for year
        
    Returns:
        Year as integer or None if not found
    """
    year_match = re.search(r'(19\d{2}|20\d{2})', text)
    if year_match:
        return int(year_match.group(0))
    return None


def extract_metadata_from_text(text: str) -> Dict[str, Any]:
    """
    Extract automotive metadata from text.
    
    Args:
        text: Text to extract metadata from
        
    Returns:
        Dictionary of extracted metadata
    """
    metadata = {}
    text_lower = text.lower()
    
    # Extract year
    year = extract_year_from_text(text)
    if year:
        metadata["year"] = year
    
    # Common manufacturers
    manufacturers = [
        "Toyota", "Honda", "Ford", "Chevrolet", "BMW", "Mercedes", "Audi", 
        "Volkswagen", "Nissan", "Hyundai", "Kia", "Subaru", "Mazda",
        "Porsche", "Ferrari", "Lamborghini", "Tesla", "Volvo", "Jaguar",
        "Land Rover", "Lexus", "Acura", "Infiniti", "Cadillac", "Jeep"
    ]
    
    # Look for manufacturer
    for manufacturer in manufacturers:
        if manufacturer.lower() in text_lower:
            metadata["manufacturer"] = manufacturer
            break
    
    # Categories
    categories = {
        "sedan": ["sedan", "saloon"],
        "suv": ["suv", "crossover"],
        "truck": ["truck", "pickup"],
        "sports": ["sports car", "supercar", "hypercar"],
        "minivan": ["minivan", "van"],
        "coupe": ["coupe", "coupé"],
        "convertible": ["convertible", "cabriolet"],
        "hatchback": ["hatchback", "hot hatch"],
        "wagon": ["wagon", "estate"],
    }
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in text_lower:
                metadata["category"] = category
                break
        if "category" in metadata:
            break
    
    # Engine types
    engine_types = {
        "gasoline": ["gasoline", "petrol", "gas"],
        "diesel": ["diesel"],
        "electric": ["electric", "ev", "battery"],
        "hybrid": ["hybrid", "phev", "plug-in"],
        "hydrogen": ["hydrogen", "fuel cell"],
    }
    
    for engine_type, keywords in engine_types.items():
        for keyword in keywords:
            if keyword in text_lower:
                metadata["engine_type"] = engine_type
                break
        if "engine_type" in metadata:
            break
    
    # Transmission types
    transmission_types = {
        "automatic": ["automatic", "auto"],
        "manual": ["manual", "stick", "speed manual"],
        "cvt": ["cvt", "continuously variable"],
        "dct": ["dct", "dual-clutch"],
    }
    
    for transmission, keywords in transmission_types.items():
        for keyword in keywords:
            if keyword in text_lower:
                metadata["transmission"] = transmission
                break
        if "transmission" in metadata:
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
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Clean the text first
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
                
                # Find the last sentence end
                sentence_ends = list(re.finditer(r'[.!?]\s+', search_text))
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
    Safely load JSON string, returning default value on error.
    
    Args:
        json_str: JSON string to parse
        default: Default value to return on error
        
    Returns:
        Parsed JSON object or default value
    """
    try:
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
