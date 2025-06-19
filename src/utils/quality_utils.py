import re
import logging
from typing import List, Tuple, Dict, Optional, Any
import jieba

logger = logging.getLogger(__name__)


def extract_key_terms(query: str) -> List[str]:
    """
    ✅ ORIGINAL: Extract key terms from query for relevance checking.
    Used by both LLM confidence scoring and document retrieval filtering.
    KEPT: jieba for proper Chinese text segmentation.
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
    ✅ ORIGINAL: Check if content contains numerical specifications in Chinese text.
    Used by both confidence scoring and document quality filtering.
    KEPT: Enhanced Chinese automotive patterns.
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
    ✅ ORIGINAL: Check if content appears garbled or corrupted.
    Used by both confidence scoring and document quality filtering.
    KEPT: Original logic for OCR error detection.
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
    ✅ ORIGINAL: Extract automotive-specific key phrases for confidence analysis.
    Used primarily by LLM confidence scoring.
    KEPT: jieba integration and Chinese error handling.
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
            '最高时速', '驱动方式', '燃油类型', '排量', '安全配置',
            '智能驾驶', '自动驾驶', '辅助驾驶', '导航系统', '语音控制'
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
                brands = ['宝马', '奔驰', '奥迪', '丰田', '本田', '大众', '特斯拉', '福特',
                          '雪佛兰', '比亚迪', '吉利', '长城', '长安', '奇瑞', '五菱']
                if any(brand in word for brand in brands):
                    technical_terms.append(word)

        return number_phrases + technical_terms

    except Exception as e:
        logger.warning(f"汽车关键词提取错误: {e}")  # Original Chinese error message
        return []


def check_acceleration_claims(text: str) -> List[str]:
    """
    ✅ ORIGINAL: Check for unrealistic acceleration claims in Chinese text.
    Used by LLM fact checker.
    KEPT: Original Chinese patterns and warning messages.
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
    ✅ ORIGINAL: Check various numerical specifications for realistic ranges in Chinese text.
    Used by LLM fact checker.
    KEPT: Original Chinese patterns and enhanced validation ranges.
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


# ==============================================================================
# NEW: POST-RERANKING FACT VALIDATION (ChatGPT's recommended approach)
# ==============================================================================

def automotive_fact_check_documents(documents: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
    """
    NEW: Perform automotive domain-specific fact validation on documents after reranking.

    This is the "Final Fact Validation Layer" that ChatGPT recommended.
    Adds warnings to document metadata without removing documents.
    Uses the original functions above for validation.

    Args:
        documents: List of (document, score) tuples after reranking

    Returns:
        Documents with automotive validation warnings added to metadata
    """
    validated_docs = []

    logger.info(f"Starting automotive fact validation for {len(documents)} documents")

    for doc, score in documents:
        content = doc.page_content
        metadata = doc.metadata.copy()

        # Initialize validation results
        validation_warnings = []
        automotive_features = {}

        try:
            # 1. Check for garbled content (basic quality)
            if has_garbled_content(content):
                validation_warnings.append("⚠️ 内容可能包含OCR识别错误")
                logger.debug(f"Document {metadata.get('id', 'unknown')} has garbled content")

            # 2. Automotive-specific validations
            if "百公里加速" in content or "0-100" in content or "acceleration" in content.lower():
                acc_warnings = check_acceleration_claims(content)
                validation_warnings.extend(acc_warnings)
                automotive_features["has_acceleration_data"] = True
                if acc_warnings:
                    logger.debug(f"Document {metadata.get('id', 'unknown')} has acceleration warnings: {acc_warnings}")

            # 3. Check numerical specifications
            spec_warnings = check_numerical_specs_realistic(content)
            validation_warnings.extend(spec_warnings)
            if spec_warnings:
                logger.debug(f"Document {metadata.get('id', 'unknown')} has spec warnings: {spec_warnings}")

            # 4. Extract automotive features for metadata (using original functions)
            automotive_features["has_numerical_data"] = has_numerical_data(content)
            automotive_features["key_phrases"] = extract_automotive_key_phrases(content)
            automotive_features["automotive_terms"] = extract_key_terms(content)
            automotive_features["automotive_terms_count"] = len(extract_key_terms(content))

            # 5. Add validation results to metadata
            if validation_warnings:
                metadata["automotive_warnings"] = validation_warnings
                metadata["validation_status"] = "has_warnings"
                logger.info(
                    f"Document {metadata.get('id', 'unknown')} flagged with {len(validation_warnings)} warnings")
            else:
                metadata["validation_status"] = "validated"

            metadata["automotive_features"] = automotive_features
            metadata["fact_checked"] = True

        except Exception as e:
            logger.error(f"Error during automotive fact checking: {e}")
            validation_warnings.append("⚠️ 验证过程出现错误")
            metadata["validation_status"] = "error"
            metadata["automotive_features"] = {}
            metadata["fact_checked"] = True

        # Create new document with enhanced metadata
        validated_doc = type(doc)(
            page_content=doc.page_content,
            metadata=metadata
        )

        validated_docs.append((validated_doc, score))

    logger.info(f"Automotive fact validation completed: {len(validated_docs)} documents processed")

    return validated_docs


def automotive_fact_check_answer(answer: str, source_documents: List[Any]) -> Dict[str, Any]:
    """
    NEW: Validate LLM answer against automotive domain knowledge.

    Used to provide fact-checking feedback on generated answers.
    Uses the original validation functions.

    Args:
        answer: Generated LLM answer
        source_documents: Documents used to generate the answer

    Returns:
        Validation results with warnings and confidence assessment
    """
    validation_results = {
        "answer_warnings": [],
        "source_warnings": [],
        "automotive_confidence": "unknown",
        "fact_check_summary": {},
    }

    logger.info("Starting automotive fact check of LLM answer")

    try:
        # 1. Check answer for obvious automotive domain errors (using original functions)
        answer_warnings = []
        answer_warnings.extend(check_acceleration_claims(answer))
        answer_warnings.extend(check_numerical_specs_realistic(answer))

        if answer_warnings:
            logger.warning(f"Answer validation found {len(answer_warnings)} issues: {answer_warnings}")

        # 2. Check if answer has automotive numerical support (using original functions)
        has_numbers = has_numerical_data(answer)
        automotive_phrases = extract_automotive_key_phrases(answer)
        automotive_terms = extract_key_terms(answer)

        logger.debug(
            f"Answer analysis - Numbers: {has_numbers}, Phrases: {len(automotive_phrases)}, Terms: {len(automotive_terms)}")

        # 3. Check source document quality
        source_warnings = []
        total_sources = len(source_documents)
        sources_with_warnings = 0

        for doc in source_documents:
            if hasattr(doc, 'metadata') and doc.metadata.get("automotive_warnings"):
                sources_with_warnings += 1
                source_warnings.extend(doc.metadata["automotive_warnings"])

        if sources_with_warnings > 0:
            logger.info(f"Source validation: {sources_with_warnings}/{total_sources} sources have warnings")

        # 4. Calculate automotive domain confidence
        confidence_factors = []

        if has_numbers:
            confidence_factors.append("有具体数值")
        if automotive_phrases:
            confidence_factors.append(f"包含{len(automotive_phrases)}个汽车专业术语")
        if automotive_terms:
            confidence_factors.append(f"识别{len(automotive_terms)}个汽车关键词")
        if not answer_warnings:
            confidence_factors.append("未发现明显错误")
        if sources_with_warnings == 0:
            confidence_factors.append("来源文档通过验证")

        # Determine confidence level
        if len(confidence_factors) >= 4 and not answer_warnings:
            automotive_confidence = "high"
        elif len(confidence_factors) >= 3 and len(answer_warnings) <= 1:
            automotive_confidence = "medium"
        else:
            automotive_confidence = "low"

        logger.info(f"Automotive confidence: {automotive_confidence} (factors: {len(confidence_factors)})")

        # 5. Compile results
        validation_results.update({
            "answer_warnings": answer_warnings,
            "source_warnings": list(set(source_warnings)),  # Remove duplicates
            "automotive_confidence": automotive_confidence,
            "fact_check_summary": {
                "has_numerical_data": has_numbers,
                "automotive_phrases_count": len(automotive_phrases),
                "automotive_terms_count": len(automotive_terms),
                "confidence_factors": confidence_factors,
                "sources_with_warnings": f"{sources_with_warnings}/{total_sources}",
            }
        })

    except Exception as e:
        logger.error(f"Error during answer fact checking: {e}")
        validation_results["automotive_confidence"] = "error"
        validation_results["answer_warnings"] = ["⚠️ 验证过程出现错误"]

    return validation_results


def format_automotive_warnings_for_user(validation_results: Dict[str, Any]) -> str:
    """
    NEW: Format automotive validation warnings as user-friendly footnotes.

    This increases user trust by showing transparent fact-checking.

    Args:
        validation_results: Results from automotive_fact_check_answer()

    Returns:
        Formatted warning text for display to users
    """
    if not validation_results.get("answer_warnings") and not validation_results.get("source_warnings"):
        return ""

    warning_text = "\n\n📋 汽车领域验证提醒:\n"

    # Answer-specific warnings
    if validation_results.get("answer_warnings"):
        warning_text += "• 答案验证:\n"
        for warning in validation_results["answer_warnings"]:
            warning_text += f"  - {warning}\n"

    # Source-specific warnings
    if validation_results.get("source_warnings"):
        warning_text += "• 来源验证:\n"
        unique_warnings = list(set(validation_results["source_warnings"]))[:3]  # Limit to 3
        for warning in unique_warnings:
            warning_text += f"  - {warning}\n"

        if len(validation_results["source_warnings"]) > 3:
            warning_text += f"  - 还有其他 {len(validation_results['source_warnings']) - 3} 项提醒...\n"

    # Confidence summary
    confidence = validation_results.get("automotive_confidence", "unknown")
    confidence_text = {
        "high": "🟢 汽车领域置信度: 高",
        "medium": "🟡 汽车领域置信度: 中等",
        "low": "🔴 汽车领域置信度: 低",
        "unknown": "⚪ 汽车领域置信度: 未知",
        "error": "❌ 汽车领域置信度: 验证错误"
    }

    warning_text += f"\n{confidence_text.get(confidence, confidence_text['unknown'])}\n"
    warning_text += "💡 建议: 重要决策前请验证多个权威来源\n"

    return warning_text


def get_automotive_validation_summary(documents: List[Any]) -> Dict[str, Any]:
    """
    NEW: Get summary of automotive validation across all documents.

    Useful for system monitoring and debugging.
    """
    total_docs = len(documents)
    docs_with_warnings = 0
    docs_with_numerical_data = 0
    all_warnings = []
    automotive_features_count = 0

    logger.debug(f"Computing validation summary for {total_docs} documents")

    for doc in documents:
        if hasattr(doc, 'metadata'):
            metadata = doc.metadata

            if metadata.get("automotive_warnings"):
                docs_with_warnings += 1
                all_warnings.extend(metadata["automotive_warnings"])

            if metadata.get("automotive_features", {}).get("has_numerical_data"):
                docs_with_numerical_data += 1

            automotive_features_count += metadata.get("automotive_features", {}).get("automotive_terms_count", 0)

    validation_quality = "high" if docs_with_warnings / total_docs < 0.2 else "medium" if docs_with_warnings / total_docs < 0.5 else "low"

    summary = {
        "total_documents": total_docs,
        "documents_with_warnings": docs_with_warnings,
        "documents_with_numerical_data": docs_with_numerical_data,
        "warning_rate": docs_with_warnings / total_docs if total_docs > 0 else 0,
        "numerical_data_rate": docs_with_numerical_data / total_docs if total_docs > 0 else 0,
        "total_warnings": len(all_warnings),
        "unique_warnings": len(set(all_warnings)),
        "automotive_features_count": automotive_features_count,
        "validation_quality": validation_quality
    }

    logger.info(
        f"Validation summary: {docs_with_warnings}/{total_docs} docs with warnings, quality: {validation_quality}")

    return summary