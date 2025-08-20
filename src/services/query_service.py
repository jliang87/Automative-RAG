"""
QueryService - Query processing logic
Called by TaskService and WorkflowService for query-related operations
Handles document retrieval, LLM inference, and response formatting
"""

import logging
import time
from typing import Dict, List, Any, Optional

from src.models.enums import QueryMode

logger = logging.getLogger(__name__)


class QueryService:
    """
    Service for query processing logic
    Handles document retrieval, LLM inference, and response processing
    """

    def __init__(self, vector_store, llm_client=None):
        self.vector_store = vector_store
        self.llm_client = llm_client

    async def retrieve_documents(self, query: str, top_k: int = 10,
                                 metadata_filter: Optional[Dict[str, Any]] = None,
                                 query_mode: str = "facts") -> Dict[str, Any]:
        """Retrieve relevant documents for a query"""

        start_time = time.time()

        try:
            logger.info(f"Retrieving documents for query: {query[:50]}...")

            # Prepare filter
            filter_dict = None
            if metadata_filter:
                filter_dict = {k: v for k, v in metadata_filter.items() if v is not None}

            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                metadata_filter=filter_dict
            )

            # Process results
            documents = []
            scores = []

            for doc, score in results:
                documents.append({
                    "content": doc.page_content,
                    "metadata": dict(doc.metadata),
                    "score": float(score)
                })
                scores.append(float(score))

            search_time = time.time() - start_time

            logger.info(f"Retrieved {len(documents)} documents in {search_time:.2f}s")

            return {
                "documents": documents,
                "scores": scores,
                "metadata": {
                    "query": query,
                    "top_k": top_k,
                    "filter_used": filter_dict,
                    "total_results": len(documents),
                    "search_time": search_time,
                    "query_mode": query_mode
                },
                "total_candidates": len(results),
                "search_time": search_time
            }

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise

    async def generate_answer(self, query: str, documents: List[Dict[str, Any]],
                              query_mode: str = "facts", template: Optional[str] = None) -> Dict[str, Any]:
        """Generate answer using LLM based on query and documents"""

        start_time = time.time()

        try:
            logger.info(f"Generating answer for query: {query[:50]}...")

            # Prepare context from documents
            context = self._prepare_context(documents, query_mode)

            # Generate prompt
            prompt = self._build_prompt(query, context, query_mode, template)

            # Generate answer (placeholder - would use actual LLM)
            answer = await self._call_llm(prompt, query_mode)

            inference_time = time.time() - start_time

            logger.info(f"Generated answer in {inference_time:.2f}s")

            return {
                "answer": answer,
                "confidence": 0.85,  # Placeholder confidence
                "reasoning": "Generated based on retrieved documents and LLM inference",
                "tokens_used": len(prompt.split()) + len(answer.split()),
                "inference_time": inference_time,
                "query_mode": query_mode
            }

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

    async def format_response(self, answer: str, documents: List[Dict[str, Any]],
                              query: str = "", response_format: str = "markdown") -> Dict[str, Any]:
        """Format the final response with sources and metadata"""

        try:
            logger.info(f"Formatting response")

            # Clean and format answer
            formatted_answer = self._clean_answer(answer)

            # Format documents for response
            formatted_documents = self._format_documents(documents)

            # Prepare sources
            sources = self._extract_sources(documents)

            # Generate analysis structure for complex modes
            analysis_structure = None
            if self._is_complex_mode(query):
                analysis_structure = self._parse_structured_answer(formatted_answer)

            return {
                "response": formatted_answer,
                "metadata": {
                    "query": query,
                    "response_format": response_format,
                    "document_count": len(documents),
                    "answer_length": len(formatted_answer),
                    "has_structure": analysis_structure is not None
                },
                "sources": sources,
                "analysis_structure": analysis_structure
            }

        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            raise

    async def post_process_response(self, answer_data: Dict[str, Any],
                                    documents: List[Dict[str, Any]], query: str,
                                    query_mode: str) -> Dict[str, Any]:
        """Post-process response with enhancements"""

        try:
            logger.info(f"Post-processing response")

            # Extract answer
            answer = answer_data.get("answer", "")

            # Apply mode-specific enhancements
            enhanced_answer = await self._enhance_answer_for_mode(answer, query_mode, documents)

            # Clean and validate documents
            clean_documents = await self._clean_documents(documents)

            # Calculate quality metrics
            quality_metrics = self._calculate_response_quality(enhanced_answer, clean_documents, query)

            return {
                "answer": enhanced_answer,
                "documents": clean_documents,
                "quality_metrics": quality_metrics,
                "query_mode": query_mode,
                "metadata": {
                    "post_processing_applied": True,
                    "enhancement_type": f"{query_mode}_specific",
                    "document_quality_score": quality_metrics.get("document_quality", 0.0)
                }
            }

        except Exception as e:
            logger.error(f"Error post-processing response: {str(e)}")
            raise

    def _prepare_context(self, documents: List[Dict[str, Any]], query_mode: str) -> str:
        """Prepare context from documents based on query mode"""

        if not documents:
            return "No relevant documents found."

        # Sort documents by score (highest first)
        sorted_docs = sorted(documents, key=lambda x: x.get("score", 0), reverse=True)

        # Select top documents based on mode
        max_docs = self._get_max_docs_for_mode(query_mode)
        selected_docs = sorted_docs[:max_docs]

        # Build context
        context_parts = []
        for i, doc in enumerate(selected_docs, 1):
            content = doc.get("content", "")
            # Truncate very long content
            if len(content) > 500:
                content = content[:500] + "..."

            context_parts.append(f"Document {i}:\n{content}")

        return "\n\n".join(context_parts)

    def _get_max_docs_for_mode(self, query_mode: str) -> int:
        """Get maximum number of documents to use based on query mode"""

        mode_limits = {
            "facts": 5,
            "features": 7,
            "tradeoffs": 8,
            "scenarios": 6,
            "debate": 10,
            "quotes": 5
        }

        return mode_limits.get(query_mode, 5)

    def _build_prompt(self, query: str, context: str, query_mode: str,
                      template: Optional[str] = None) -> str:
        """Build prompt for LLM based on query mode"""

        if template:
            return template.format(query=query, context=context)

        # Mode-specific prompts
        mode_prompts = {
            "facts": f"""基于以下文档，回答用户的问题。请提供准确、具体的信息。

文档内容：
{context}

用户问题：{query}

请提供准确的答案：""",

            "features": f"""基于以下文档，分析是否应该添加用户提到的功能。请从多个角度进行评估。

文档内容：
{context}

用户问题：{query}

请从以下角度分析：
**优势：**
- 

**考虑因素：**
- 

**建议：**""",

            "tradeoffs": f"""基于以下文档，分析用户提到的选择的利弊。请提供平衡的分析。

文档内容：
{context}

用户问题：{query}

请提供利弊分析：
**优点：**
- 

**缺点：**
- 

**结论：**""",

            "scenarios": f"""基于以下文档，分析用户提到的场景中的表现。请考虑实际使用情况。

文档内容：
{context}

用户问题：{query}

请分析不同场景下的表现：""",

            "debate": f"""基于以下文档，从多个角色的角度讨论用户的问题。

文档内容：
{context}

用户问题：{query}

请从以下角度讨论：
**技术角度：**

**产品角度：**

**用户角度：**

**综合结论：**""",

            "quotes": f"""基于以下文档，提取与用户问题相关的用户评价和反馈。

文档内容：
{context}

用户问题：{query}

请提取相关的用户评论："""
        }

        return mode_prompts.get(query_mode, mode_prompts["facts"])

    async def _call_llm(self, prompt: str, query_mode: str) -> str:
        """Call LLM to generate answer (placeholder implementation)"""

        # Placeholder implementation - would call actual LLM
        logger.info(f"Calling LLM for {query_mode} mode")

        # Simulate LLM response based on mode
        if query_mode == "facts":
            return "根据提供的文档，这里是相关的技术规格和参数信息。具体数值需要查阅最新的官方文档进行确认。"
        elif query_mode == "features":
            return """**优势：**
- 可以提升用户体验
- 增加产品竞争力

**考虑因素：**
- 实施成本和技术复杂度
- 用户接受度和市场需求

**建议：**
建议进行用户调研和技术可行性评估后再做决定。"""
        elif query_mode == "tradeoffs":
            return """**优点：**
- 性能提升
- 用户体验改善

**缺点：**
- 成本增加
- 实施复杂度提高

**结论：**
需要综合考虑成本效益比和长期价值。"""
        elif query_mode == "scenarios":
            return """**日常使用场景：**
在城市通勤中表现良好，操作便捷。

**特殊场景：**
在恶劣天气或高负载情况下可能需要额外注意。

**整体评估：**
大部分场景下都能满足用户需求。"""
        elif query_mode == "debate":
            return """**技术角度：**
从技术实现角度看，方案可行但需要考虑技术债务。

**产品角度：**
符合产品路线图，能够增强市场竞争力。

**用户角度：**
用户反馈积极，能够解决实际痛点。

**综合结论：**
建议推进，但需要制定详细的实施计划。"""
        elif query_mode == "quotes":
            return """"这个功能很实用，大大提升了使用体验。"

"虽然有学习成本，但掌握后很方便。"

"希望后续版本能够进一步优化。"

*注：以上为用户反馈摘录*"""
        else:
            return "基于提供的文档，这里是相关信息的综合分析。"

    def _clean_answer(self, answer: str) -> str:
        """Clean and format answer text"""

        if not answer:
            return ""

        # Remove extra whitespace
        answer = " ".join(answer.split())

        # Remove common artifacts
        answer = answer.replace("答案：", "").replace("回答：", "")

        # Ensure proper ending
        if answer and not answer.endswith(("。", "！", "？", ".", "!", "?")):
            answer += "。"

        return answer.strip()

    def _format_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format documents for response"""

        formatted_docs = []

        for doc in documents:
            content = doc.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."

            formatted_docs.append({
                "id": doc.get("metadata", {}).get("id", ""),
                "content": content,
                "metadata": doc.get("metadata", {}),
                "score": doc.get("score", 0.0),
                "relevance_score": doc.get("score", 0.0)
            })

        return formatted_docs

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from documents"""

        sources = []
        seen_sources = set()

        for doc in documents:
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "unknown")

            if source not in seen_sources:
                sources.append({
                    "source": source,
                    "title": metadata.get("title", ""),
                    "url": metadata.get("url", ""),
                    "type": metadata.get("source_type", "document")
                })
                seen_sources.add(source)

        return sources

    def _is_complex_mode(self, query: str) -> bool:
        """Check if query requires complex structured response"""

        complex_indicators = [
            "分析", "对比", "利弊", "优缺点", "场景", "讨论", "评估"
        ]

        return any(indicator in query for indicator in complex_indicators)

    def _parse_structured_answer(self, answer: str) -> Optional[Dict[str, Any]]:
        """Parse structured answer for complex modes"""

        if not answer:
            return None

        # Simple structure parsing
        sections = []
        lines = answer.split('\n')
        current_section = None
        current_content = []

        for line in lines:
            line = line.strip()
            if line.startswith('**') and line.endswith('**'):
                # Save previous section
                if current_section and current_content:
                    sections.append({
                        "title": current_section,
                        "content": " ".join(current_content)
                    })

                # Start new section
                current_section = line.strip('*')
                current_content = []
            elif line:
                current_content.append(line)

        # Add last section
        if current_section and current_content:
            sections.append({
                "title": current_section,
                "content": " ".join(current_content)
            })

        if sections:
            return {
                "summary": answer.split('\n')[0] if answer else "",
                "main_points": [s["title"] for s in sections],
                "detailed_sections": sections,
                "total_sections": len(sections)
            }

        return None

    async def _enhance_answer_for_mode(self, answer: str, query_mode: str,
                                       documents: List[Dict[str, Any]]) -> str:
        """Apply mode-specific enhancements to answer"""

        if not answer:
            return answer

        # Add confidence indicators for facts mode
        if query_mode == "facts":
            if len(documents) < 3:
                answer += "\n\n*注：基于有限的文档信息，建议进一步验证。*"
            elif all(doc.get("score", 0) < 0.7 for doc in documents):
                answer += "\n\n*注：相关性较低，建议查阅官方资料确认。*"

        # Add balanced perspective note for debate mode
        elif query_mode == "debate":
            if "综合结论" not in answer:
                answer += "\n\n*注：以上观点仅供参考，实际决策需考虑更多因素。*"

        # Add source diversity note for quotes mode
        elif query_mode == "quotes":
            unique_sources = len(set(doc.get("metadata", {}).get("source", "") for doc in documents))
            if unique_sources > 1:
                answer += f"\n\n*基于 {unique_sources} 个不同来源的用户反馈*"

        return answer

    async def _clean_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and validate documents"""

        cleaned_docs = []

        for doc in documents:
            content = doc.get("content", "")

            # Skip documents with very short content
            if len(content.strip()) < 20:
                continue

            # Skip documents with very low scores
            if doc.get("score", 0) < 0.1:
                continue

            # Clean content
            content = content.strip()
            if len(content) > 1000:
                content = content[:1000] + "..."

            cleaned_doc = {
                "id": doc.get("metadata", {}).get("id", ""),
                "content": content,
                "metadata": doc.get("metadata", {}),
                "score": doc.get("score", 0.0),
                "relevance_score": doc.get("score", 0.0)
            }

            cleaned_docs.append(cleaned_doc)

        # Sort by relevance score (highest first)
        cleaned_docs.sort(key=lambda x: x.get("score", 0), reverse=True)

        return cleaned_docs

    def _calculate_response_quality(self, answer: str, documents: List[Dict[str, Any]],
                                    query: str) -> Dict[str, Any]:
        """Calculate quality metrics for the response"""

        quality_metrics = {
            "answer_length": len(answer),
            "word_count": len(answer.split()),
            "document_count": len(documents),
            "avg_document_score": 0.0,
            "has_structure": "**" in answer or "#" in answer,
            "completeness_score": 0.0,
            "relevance_score": 0.0
        }

        # Calculate average document score
        if documents:
            total_score = sum(doc.get("score", 0) for doc in documents)
            quality_metrics["avg_document_score"] = total_score / len(documents)

        # Calculate completeness score based on answer length and structure
        word_count = quality_metrics["word_count"]
        if word_count >= 50:
            quality_metrics["completeness_score"] = min(1.0, word_count / 200)
        else:
            quality_metrics["completeness_score"] = word_count / 50

        # Calculate relevance score based on query keywords in answer
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        if query_words:
            overlap = len(query_words.intersection(answer_words))
            quality_metrics["relevance_score"] = overlap / len(query_words)

        # Calculate overall quality score
        scores = [
            quality_metrics["completeness_score"],
            quality_metrics["relevance_score"],
            quality_metrics["avg_document_score"],
            1.0 if quality_metrics["has_structure"] else 0.5
        ]

        quality_metrics["overall_quality"] = sum(scores) / len(scores)
        quality_metrics["document_quality"] = quality_metrics["avg_document_score"]

        return quality_metrics

    def get_query_processing_stats(self) -> Dict[str, Any]:
        """Get query processing statistics"""

        # Placeholder implementation
        return {
            "total_queries_processed": 0,
            "average_response_time": 0.0,
            "mode_distribution": {
                "facts": 0,
                "features": 0,
                "tradeoffs": 0,
                "scenarios": 0,
                "debate": 0,
                "quotes": 0
            },
            "average_document_count": 0.0,
            "success_rate": 100.0
        }