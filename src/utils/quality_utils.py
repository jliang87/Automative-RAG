import re
import logging
from typing import List, Tuple, Dict, Optional
import jieba

logger = logging.getLogger(__name__)


def extract_key_terms(query: str) -> List[str]:
    """
    Extract key terms from query for relevance checking.
    Used by both LLM confidence scoring and document retrieval filtering.
    """
    # Remove common Chinese stop words
    stop_words = {
        "的", "是", "在", "有", "和", "与", "及", "或", "但", "而", "了", "吗", "呢",
        "什么", "如何", "怎么", "多少", "哪个", "哪些", "这个", "那个", "一个",
        "可以", "能够", "应该", "会", "将", "把", "被", "让", "使", "为了"
    }

    # Use jieba for Chinese text segmentation
    terms = jieba.lcut(query.lower())

    # Filter terms: remove stop words and keep meaningful terms
    key_terms = [term for term in terms if len(term) > 1 and term not in stop_words]

    return key_terms


def has_numerical_data(content: str) -> bool:
    """
    Check if content contains numerical specifications in Chinese text.
    Used by both confidence scoring and document quality filtering.
    """
    # Enhanced patterns for Chinese automotive specifications
    number_patterns = [
        # Basic units with Chinese and English
        r'\d+\.?\d*\s*(?:秒|升|L|马力|HP|牛米|Nm|公里|km|米|m|毫米|mm|公斤|kg|元|万元|千克)',
        r'\d+\.?\d*\s*(?:seconds?|liters?|horsepower|torque|kilometers?|meters?|kilograms?|minutes?)',

        # Years
        r'\d{4}年',

        # Decimal measurements
        r'\d+\.\d+[Ll升]',

        # Power measurements
        r'\d+[Ww瓦]',
        r'\d+\s*(?:千瓦|kW|KW)',

        # Chinese number units
        r'\d+\s*(?:万|千|百)',

        # Automotive specific Chinese terms
        r'\d+\.?\d*\s*(?:马力|扭矩|排量|功率)',

        # Speed measurements
        r'\d+\s*(?:公里/小时|千米/小时|km/h|kmh)',

        # Capacity measurements
        r'\d+\s*(?:立方|立方米|立方厘米)',

        # Weight measurements
        r'\d+\s*(?:吨|公斤|千克|斤)',

        # Dimension measurements
        r'\d+\s*(?:毫米|厘米|分米|米)',

        # Price measurements
        r'\d+\s*(?:元|万元|千元)',

        # Fuel consumption
        r'\d+\.?\d*\s*(?:升/百公里|L/100km)',

        # Time measurements for automotive specs
        r'\d+\.?\d*\s*(?:分钟|小时|秒钟)',

        # Performance ratios
        r'\d+\.?\d*\s*(?:比|倍)',
    ]

    for pattern in number_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True

    return False


def has_garbled_content(content: str) -> bool:
    """
    Check if content appears garbled or corrupted.
    Used by both confidence scoring and document quality filtering.
    """
    if len(content) < 10:
        return True

    # Check for excessive special characters (reasonable threshold)
    special_char_ratio = sum(
        1 for c in content
        if not c.isalnum() and not c.isspace() and c not in '.,!?;:()[]{}"-'
    ) / len(content)

    if special_char_ratio > 0.4:  # 40% special characters
        return True

    # Check for repeated patterns that suggest OCR errors
    if re.search(r'(.)\1{8,}', content):  # Same character repeated 8+ times
        return True

    return False


def extract_automotive_key_phrases(text: str) -> List[str]:
    """
    Extract automotive-specific key phrases for confidence analysis.
    Used primarily by LLM confidence scoring.
    FIXED: Better error handling with Chinese messages.
    """
    try:
        # Extract numbers with automotive units as key phrases
        number_phrases = re.findall(
            r'\d+\.?\d*\s*(?:秒|升|马力|牛米|公里|米|毫米|公斤|元|km/h|mph|HP)',
            text,
            re.IGNORECASE
        )

        # Add automotive-specific words to jieba dictionary
        automotive_terms = [
            '百公里加速', '后备箱', '行李箱', '尾箱', '马力', '扭矩', '牛米',
            '油耗', '续航', '充电', '变速箱', '发动机', '底盘', '悬架',
            '轴距', '车长', '车宽', '车高', '整备质量', '最大功率',
            '最高时速', '驱动方式', '燃油类型', '排量'
        ]

        for term in automotive_terms:
            jieba.add_word(term)

        # Extract technical terms using jieba
        technical_terms = []
        words = jieba.lcut(text)
        for word in words:
            # Keep automotive technical terms
            if word in automotive_terms:
                technical_terms.append(word)
            # Keep words with numbers (like model names, years)
            elif len(word) > 2 and any(char.isdigit() for char in word):
                technical_terms.append(word)
            # Keep brand/model names (Chinese characters, length > 1)
            elif len(word) > 1 and all('\u4e00' <= char <= '\u9fff' for char in word):
                # Check if it might be a car brand or model
                brands = ['宝马', '奔驰', '奥迪', '丰田', '本田', '大众', '特斯拉', '福特', '雪佛兰']
                if any(brand in word for brand in brands):
                    technical_terms.append(word)

        return number_phrases + technical_terms

    except Exception as e:
        logger.warning(f"汽车关键词提取错误: {e}")  # FIXED: Chinese error message
        return []


def check_acceleration_claims(text: str) -> List[str]:
    """
    Check for unrealistic acceleration claims in Chinese text.
    Used by LLM fact checker.
    ALREADY OPTIMIZED: Uses Chinese patterns and warning messages.
    """
    warnings = []

    # Realistic range for 0-100 km/h acceleration
    min_acceleration, max_acceleration = 2.0, 20.0

    # Enhanced patterns for Chinese text with various formats
    patterns = [
        r'(?:百公里加速|0-100|零至100|0到100|加速时间|加速性能).*?(\d+\.?\d*)\s*秒',
        r'(\d+\.?\d*)\s*秒.*?(?:百公里加速|0-100|零至100|加速时间)',
        r'(?:从|由).*?(?:0|零).*?(?:到|至).*?(?:100|一百).*?(?:公里|千米).*?(\d+\.?\d*)\s*秒',
        r'加速.*?(\d+\.?\d*)\s*秒.*?(?:百公里|100公里)',
        r'(?:静止|起步).*?(?:到|至).*?100.*?(\d+\.?\d*)\s*秒',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                value = float(match)

                if value < min_acceleration:
                    warnings.append(f"⚠️ 可疑的加速数据: {value}秒 (过快，现实中不可能)")
                elif value > max_acceleration:
                    warnings.append(f"⚠️ 可疑的加速数据: {value}秒 (过慢，不符合常理)")

            except ValueError:
                continue

    return warnings


def check_numerical_specs_realistic(text: str) -> List[str]:
    """
    Check various numerical specifications for realistic ranges in Chinese text.
    Used by LLM fact checker.
    ALREADY OPTIMIZED: Uses Chinese patterns and warning messages.
    """
    warnings = []

    # Define realistic ranges for automotive specs
    spec_ranges = {
        "horsepower": (50, 2000),  # HP
        "trunk_capacity": (100, 2000),  # Liters
        "top_speed": (120, 400),  # km/h
        "fuel_consumption": (3.0, 25.0),  # L/100km
        "torque": (50, 2000),  # Nm
        "displacement": (0.5, 8.0),  # Liters
        "weight": (800, 5000),  # kg
        "wheelbase": (2000, 4000),  # mm
        "price": (50000, 5000000),  # CNY
    }

    # Enhanced Chinese patterns for horsepower/power
    hp_patterns = [
        r'(?:最大功率|功率|马力|输出功率).*?(\d+)\s*(?:马力|HP|hp|匹)',
        r'(\d+)\s*(?:马力|HP|hp|匹).*?(?:功率|输出|最大)',
        r'功率.*?(\d+)\s*(?:千瓦|kW|KW)',
        r'(\d+)\s*(?:千瓦|kW|KW).*?功率',
    ]

    for pattern in hp_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                value = float(match)
                # Convert kW to HP if needed (1 kW ≈ 1.34 HP)
                if '千瓦' in pattern or 'kW' in pattern or 'KW' in pattern:
                    value = value * 1.34

                min_val, max_val = spec_ranges["horsepower"]
                if not (min_val <= value <= max_val):
                    warnings.append(f"⚠️ 可疑的功率数据: {value:.0f}马力 (超出合理范围)")
            except ValueError:
                continue

    # Enhanced Chinese patterns for trunk capacity
    trunk_patterns = [
        r'(?:后备箱|行李箱|尾箱|储物箱|载物).*?(?:容积|空间|体积|容量).*?(\d+)\s*(?:升|L|l|立方)',
        r'(?:容积|空间|体积|容量).*?(\d+)\s*(?:升|L|l).*?(?:后备箱|行李箱|尾箱)',
        r'(?:后备箱|尾箱).*?(\d+)\s*(?:升|L|l)',
        r'储物空间.*?(\d+)\s*(?:升|L|l)',
    ]

    for pattern in trunk_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                value = float(match)
                min_val, max_val = spec_ranges["trunk_capacity"]
                if not (min_val <= value <= max_val):
                    warnings.append(f"⚠️ 可疑的后备箱容积: {value}升 (超出合理范围)")
            except ValueError:
                continue

    # Enhanced Chinese patterns for top speed
    speed_patterns = [
        r'(?:最高时速|最大速度|极速|顶速).*?(\d+)\s*(?:公里|千米|km/h|kmh)',
        r'时速.*?(?:最高|最大|可达).*?(\d+)\s*(?:公里|千米|km/h)',
        r'速度.*?(\d+)\s*(?:公里|千米|km/h).*?(?:最高|最大)',
    ]

    for pattern in speed_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                value = float(match)
                min_val, max_val = spec_ranges["top_speed"]
                if not (min_val <= value <= max_val):
                    warnings.append(f"⚠️ 可疑的最高时速: {value}公里/小时 (超出合理范围)")
            except ValueError:
                continue

    # Enhanced Chinese patterns for fuel consumption
    fuel_patterns = [
        r'(?:油耗|燃油消耗|百公里油耗).*?(\d+\.?\d*)\s*(?:升|L|l)',
        r'(\d+\.?\d*)\s*(?:升|L|l).*?(?:百公里|油耗|燃油)',
        r'综合油耗.*?(\d+\.?\d*)',
        r'工信部油耗.*?(\d+\.?\d*)',
    ]

    for pattern in fuel_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                value = float(match)
                min_val, max_val = spec_ranges["fuel_consumption"]
                if not (min_val <= value <= max_val):
                    warnings.append(f"⚠️ 可疑的油耗数据: {value}升/100公里 (超出合理范围)")
            except ValueError:
                continue

    return warnings