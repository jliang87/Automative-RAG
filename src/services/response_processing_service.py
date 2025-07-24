import logging
import re
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class ResponseProcessingService:
    """Service for processing and formatting responses."""

    def __init__(self):
        self.max_answer_length = 2000
        self.min_answer_length = 50

    def clean_answer(self, answer: str) -> str:
        """Clean and format the answer text."""
        if not answer:
            return ""

        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer.strip())

        # Remove any markdown artifacts
        answer = re.sub(r'```[a-zA-Z]*\n', '', answer)
        answer = re.sub(r'```\n?', '', answer)

        # Clean up common LLM artifacts
        answer = re.sub(r'^(答案|回答|Answer)[:：]\s*', '', answer)
        answer = re.sub(r'根据提供的文档[，,]\s*', '', answer)
        answer = re.sub(r'基于[上述|以上]*[文档|资料|信息][，,]\s*', '', answer)

        # Remove repeated phrases
        answer = re.sub(r'(.{10,}?)\1+', r'\1', answer)

        # Ensure proper spacing around Chinese punctuation
        answer = re.sub(r'([。！？])([^\s])', r'\1 \2', answer)
        answer = re.sub(r'([，、；])([^\s])', r'\1 \2', answer)

        # Remove excessive punctuation
        answer = re.sub(r'[。]{2,}', '。', answer)
        answer = re.sub(r'[！]{2,}', '！', answer)
        answer = re.sub(r'[？]{2,}', '？', answer)

        # Trim to reasonable length
        if len(answer) > self.max_answer_length:
            # Find a good breaking point
            truncate_point = self.max_answer_length
            for punct in ['。', '！', '？', '\n']:
                last_punct = answer.rfind(punct, 0, self.max_answer_length)
                if last_punct > self.max_answer_length * 0.8:  # If we find punctuation in the last 20%
                    truncate_point = last_punct + 1
                    break

            answer = answer[:truncate_point].strip()
            if not answer.endswith(('。', '！', '？')):
                answer += "..."

        return answer.strip()

    def parse_structured_answer(self, answer: str, mode: str) -> Optional[Dict[str, Any]]:
        """Parse structured answer for two-layer modes."""
        if not answer or mode not in ['features', 'tradeoffs', 'scenarios']:
            return None

        try:
            # Attempt to parse structured sections
            sections = []
            current_section = None
            current_content = []

            lines = answer.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if line is a section header
                is_header = self._is_section_header(line, mode)

                if is_header and len(line) < 100:  # Likely a header
                    # Save previous section
                    if current_section and current_content:
                        sections.append({
                            "title": current_section,
                            "content": " ".join(current_content).strip()
                        })

                    # Start new section
                    current_section = line
                    current_content = []
                else:
                    # Add to current content
                    if line:  # Only add non-empty lines
                        current_content.append(line)

            # Add last section
            if current_section and current_content:
                sections.append({
                    "title": current_section,
                    "content": " ".join(current_content).strip()
                })

            # Extract summary and conclusion
            paragraphs = [s for s in answer.split('\n\n') if s.strip()]
            summary = paragraphs[0] if paragraphs else ""
            conclusion = paragraphs[-1] if len(paragraphs) > 1 else ""

            # Generate main points from sections
            main_points = []
            for section in sections:
                # Extract first sentence or key point from each section
                content = section["content"]
                first_sentence = content.split('。')[0] + '。' if '。' in content else content[:100] + "..."
                main_points.append(first_sentence)

            structure = {
                "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                "main_points": main_points[:5],  # Limit to 5 main points
                "detailed_sections": sections,
                "conclusion": conclusion[:200] + "..." if len(conclusion) > 200 else conclusion,
                "total_sections": len(sections),
                "estimated_reading_time": max(1, len(answer.split()) // 200),  # minutes
                "structure_quality": self._assess_structure_quality(sections, mode)
            }

            logger.info(f"Parsed structured answer with {len(sections)} sections for mode {mode}")
            return structure

        except Exception as e:
            logger.error(f"Error parsing structured answer: {e}")
            return None

    def _is_section_header(self, line: str, mode: str) -> bool:
        """Check if a line is likely a section header based on mode."""
        line_lower = line.lower()

        # Common header patterns for different modes
        mode_keywords = {
            'features': ['优点', '缺点', '功能', '特性', '建议', '分析', '评估', '优势', '劣势'],
            'tradeoffs': ['优点', '缺点', '利弊', '对比', '优势', '劣势', '权衡', '分析', '结论'],
            'scenarios': ['场景', '情况', '使用', '环境', '条件', '表现', '性能', '体验']
        }

        keywords = mode_keywords.get(mode, [])

        # Check for keyword presence
        has_keyword = any(keyword in line for keyword in keywords)

        # Check for header formatting patterns
        has_formatting = any([
            line.startswith(('**', '##', '#')),  # Markdown headers
            line.endswith(':') or line.endswith('：'),  # Colon endings
            re.match(r'^\d+\.\s', line),  # Numbered lists
            re.match(r'^[一二三四五六七八九十]+[、.]', line),  # Chinese numbers
            len(line.split()) <= 5 and has_keyword  # Short keyword lines
        ])

        return has_keyword and (has_formatting or len(line) < 50)

    def _assess_structure_quality(self, sections: List[Dict[str, str]], mode: str) -> Dict[str, Any]:
        """Assess the quality of the parsed structure."""
        if not sections:
            return {"score": 0.0, "issues": ["No sections found"]}

        score = 0.0
        issues = []

        # Check number of sections
        if len(sections) >= 2:
            score += 0.3
        else:
            issues.append("Too few sections for structured analysis")

        # Check section content length
        avg_content_length = sum(len(s["content"]) for s in sections) / len(sections)
        if avg_content_length > 50:
            score += 0.3
        else:
            issues.append("Sections have insufficient content")

        # Check title quality
        good_titles = sum(1 for s in sections if len(s["title"]) > 3 and len(s["title"]) < 50)
        if good_titles == len(sections):
            score += 0.2
        else:
            issues.append("Some section titles are not well-formed")

        # Mode-specific checks
        if mode == 'tradeoffs':
            has_pros = any('优' in s["title"] or '利' in s["title"] for s in sections)
            has_cons = any('缺' in s["title"] or '弊' in s["title"] for s in sections)
            if has_pros and has_cons:
                score += 0.2
            else:
                issues.append("Missing pros/cons structure for tradeoffs analysis")

        return {"score": min(score, 1.0), "issues": issues}

    def format_for_mode(self, answer: str, mode: str, metadata: Dict[str, Any] = None) -> str:
        """Format answer based on query mode."""
        if not answer:
            return ""

        cleaned_answer = self.clean_answer(answer)

        if mode == "facts":
            # Add factual confidence indicators
            if metadata and metadata.get("confidence", 0) < 0.7:
                cleaned_answer += "\n\n*注：此信息的置信度较低，建议进一步验证。*"

            # Add source count if available
            if metadata and metadata.get("source_count"):
                cleaned_answer += f"\n\n*基于 {metadata['source_count']} 个信息源*"

        elif mode in ["features", "tradeoffs", "scenarios"]:
            # Enhance structure for complex modes
            if not any(marker in cleaned_answer for marker in ['**', '##', '优点', '缺点']):
                # Add basic structure if missing
                sentences = [s.strip() for s in cleaned_answer.split('。') if s.strip()]
                if len(sentences) >= 4:
                    middle = len(sentences) // 2
                    structured = "**主要分析：**\n" + "。".join(sentences[:middle]) + "。\n\n"
                    structured += "**进一步考虑：**\n" + "。".join(sentences[middle:]) + "。"
                    cleaned_answer = structured

        elif mode == "quotes":
            # Format quotes with better visual separation
            if '"' in cleaned_answer or '"' in cleaned_answer or '说' in cleaned_answer:
                # Improve quote formatting
                cleaned_answer = re.sub(r'(["""])', r'\n\1', cleaned_answer)
                cleaned_answer = re.sub(r'\n+', '\n\n', cleaned_answer)
                cleaned_answer = cleaned_answer.strip()

        elif mode == "debate":
            # Add role indicators if not present
            if not any(role in cleaned_answer for role in ['工程师', '产品经理', '用户', '观点']):
                # Add basic debate structure
                sentences = [s.strip() for s in cleaned_answer.split('。') if s.strip()]
                if len(sentences) >= 3:
                    formatted = "**技术角度：** " + sentences[0] + "。\n\n"
                    if len(sentences) > 1:
                        formatted += "**产品角度：** " + sentences[1] + "。\n\n"
                    if len(sentences) > 2:
                        formatted += "**用户角度：** " + "。".join(sentences[2:]) + "。"
                    cleaned_answer = formatted

        return cleaned_answer

    def enhance_answer_quality(self, answer: str, documents: List[Dict[str, Any]],
                               query: str, mode: str) -> str:
        """Enhance answer quality based on available documents and query context."""
        if not answer:
            return ""

        enhanced_answer = answer

        # Add document-based enhancements
        if documents:
            doc_count = len(documents)
            avg_relevance = sum(doc.get("score", 0) for doc in documents) / doc_count

            # Add confidence footer based on document quality
            if avg_relevance > 0.8:
                confidence_note = f"\n\n*高置信度答案 - 基于 {doc_count} 个高相关性文档*"
            elif avg_relevance > 0.6:
                confidence_note = f"\n\n*中等置信度答案 - 基于 {doc_count} 个相关文档*"
            else:
                confidence_note = f"\n\n*基础答案 - 基于 {doc_count} 个文档，建议进一步验证*"

            enhanced_answer += confidence_note

        # Add query-specific enhancements
        if len(query) > 50:  # Complex query
            if mode == "facts" and "规格" in query:
                enhanced_answer += "\n\n*提示：具体规格参数可能因车型年份和配置而异*"
            elif mode == "features" and "建议" in query:
                enhanced_answer += "\n\n*建议：最终决策请结合具体使用需求和预算考虑*"

        return enhanced_answer

    def generate_response_metadata(self, answer: str, mode: str,
                                   processing_stats: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate metadata about the response."""
        metadata = {
            "answer_length": len(answer),
            "word_count": len(answer.split()),
            "sentence_count": len([s for s in answer.split('。') if s.strip()]),
            "mode": mode,
            "estimated_reading_time_seconds": max(30, len(answer.split()) * 0.5),  # ~2 words per second
            "complexity_level": self._assess_answer_complexity(answer, mode),
            "quality_indicators": self._generate_quality_indicators(answer, mode)
        }

        if processing_stats:
            metadata["processing_stats"] = processing_stats

        return metadata

    def _assess_answer_complexity(self, answer: str, mode: str) -> str:
        """Assess the complexity level of the answer."""
        word_count = len(answer.split())
        sentence_count = len([s for s in answer.split('。') if s.strip()])

        # Mode-based complexity adjustment
        mode_complexity = {
            "facts": 0,
            "quotes": 0,
            "features": 1,
            "scenarios": 1,
            "tradeoffs": 2,
            "debate": 2
        }

        base_complexity = mode_complexity.get(mode, 0)

        if word_count < 50:
            complexity_score = 0 + base_complexity
        elif word_count < 150:
            complexity_score = 1 + base_complexity
        else:
            complexity_score = 2 + base_complexity

        if complexity_score <= 1:
            return "simple"
        elif complexity_score <= 3:
            return "moderate"
        else:
            return "complex"

    def _generate_quality_indicators(self, answer: str, mode: str) -> Dict[str, Any]:
        """Generate quality indicators for the answer."""
        indicators = {
            "has_structure": bool(re.search(r'[**#]|\n\n', answer)),
            "has_examples": "例如" in answer or "比如" in answer,
            "has_citations": "根据" in answer or "基于" in answer,
            "appropriate_length": 50 <= len(answer) <= 1000,
            "clear_conclusion": answer.strip().endswith(('。', '！', '？')),
            "uses_chinese": bool(re.search(r'[\u4e00-\u9fff]', answer))
        }

        # Mode-specific indicators
        if mode == "tradeoffs":
            indicators["has_pros_cons"] = any(word in answer for word in ['优点', '缺点', '利', '弊'])
        elif mode == "scenarios":
            indicators["has_scenarios"] = any(word in answer for word in ['场景', '情况', '使用'])
        elif mode == "debate":
            indicators["has_multiple_perspectives"] = any(word in answer for word in ['观点', '角度', '认为'])

        # Calculate overall quality score
        quality_score = sum(indicators.values()) / len(indicators)
        indicators["overall_quality"] = quality_score

        return indicators