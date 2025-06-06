import re
from typing import Any, Dict, Optional

__all__ = [
    "clean_text",
    "extract_year_from_text",
    "extract_metadata_from_text",
]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)

    # Strip whitespace from beginning and end
    text = text.strip()

    # Replace common Unicode characters that might still appear
    replacements = {
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201c': '"',  # Left double quotation mark
        '\u201d': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '--',  # Em dash
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
    if not isinstance(text, str):
        return None

    year_match = re.search(r'(19\d{2}|20\d{2})', text)
    if year_match:
        return int(year_match.group(0))
    return None


def extract_metadata_from_text(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {}

    metadata = {}
    text_lower = text.lower()

    # Extract year
    year = extract_year_from_text(text)
    if year:
        metadata["year"] = year

    # Chinese and English manufacturer detection
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
            metadata["manufacturer"] = chinese_name
            break
        # Check for English name (case insensitive)
        elif english_name.lower() in text_lower:
            metadata["manufacturer"] = english_name
            break

    # Chinese category detection
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

    # Chinese engine type detection
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

    # Chinese transmission type detection
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

    # Extract Chinese model names (common patterns)
    if "manufacturer" in metadata:
        manufacturer = metadata["manufacturer"]

        # Common Chinese model patterns
        model_patterns = {
            "宝马": [r"(X[1-7]|[1-8]系|i[3-8]|Z4)", r"(X[1-7]|Series [1-8]|i[3-8]|Z4)"],
            "奔驰": [r"([A-Z]级|[A-Z]-Class|GLA|GLC|GLE|GLS|AMG)", r"([A-Z]级|[A-Z]-Class|GLA|GLC|GLE|GLS|AMG)"],
            "奥迪": [r"(A[1-8]|Q[2-8]|TT|R8)", r"(A[1-8]|Q[2-8]|TT|R8)"],
            "丰田": [r"(凯美瑞|卡罗拉|汉兰达|普拉多|陆地巡洋舰)", r"(Camry|Corolla|Highlander|Prado|Land Cruiser)"],
            "本田": [r"(雅阁|思域|CR-V|奥德赛)", r"(Accord|Civic|CR-V|Odyssey)"],
        }

        if manufacturer in model_patterns:
            for pattern in model_patterns[manufacturer]:
                model_match = re.search(pattern, text, re.IGNORECASE)
                if model_match:
                    metadata["model"] = model_match.group(1)
                    break

    return metadata