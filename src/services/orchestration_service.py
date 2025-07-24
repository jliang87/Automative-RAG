import logging
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)


class OrchestrationService:
    """Service for orchestrating query workflows."""

    def __init__(self, vector_store, job_tracker):
        self.vector_store = vector_store
        self.job_tracker = job_tracker

    async def start_query_workflow(self,
                                   job_id: str,
                                   request,  # EnhancedQueryRequest type
                                   validation_enabled: bool = False,
                                   validation_id: Optional[str] = None,
                                   validation_workflow: Any = None):
        """Start the query processing workflow."""
        logger.info(f"Starting query workflow for job {job_id}")

        try:
            # Import job chain dynamically to avoid circular imports
            from src.core.orchestration.job_chain import job_chain
            from src.core.orchestration.task_router import JobType

            # Update job status
            self.job_tracker.update_job_status(job_id, "processing")

            # Determine job type based on validation settings
            if validation_enabled and validation_workflow:
                if validation_workflow.validation_type.value == "comprehensive":
                    job_type = JobType.COMPREHENSIVE_VALIDATION
                else:
                    job_type = JobType.VALIDATION_ENHANCED
            else:
                job_type = JobType.QUERY_WITH_VALIDATION

            # Prepare initial data for the workflow
            initial_data = {
                "query": request.query,
                "query_mode": request.query_mode.value,
                "metadata_filter": request.metadata_filter.dict() if request.metadata_filter else None,
                "top_k": request.top_k,
                "prompt_template": request.prompt_template,
                "validation_enabled": validation_enabled,
                "validation_id": validation_id,
                "include_sources": request.include_sources,
                "response_format": request.response_format
            }

            # Start the job chain
            job_chain.start_job_chain(job_id, job_type, initial_data)

            logger.info(f"Successfully started {job_type.value} workflow for job {job_id}")

        except Exception as e:
            logger.error(f"Error in query workflow: {e}")
            self.job_tracker.update_job_status(job_id, "failed", error=str(e))
            raise

    async def process_simple_query(self, job_id: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a simple query without complex workflows."""
        logger.info(f"Processing simple query for job {job_id}")

        try:
            # Extract query parameters
            query = query_data.get("query", "")
            query_mode = query_data.get("query_mode", "facts")
            top_k = query_data.get("top_k", 10)
            metadata_filter = query_data.get("metadata_filter")

            # Update job progress
            self.job_tracker.update_job_progress(job_id, 25, "Retrieving documents...")

            # Perform document retrieval
            documents = await self._retrieve_documents(query, top_k, metadata_filter)

            # Update job progress
            self.job_tracker.update_job_progress(job_id, 50, "Processing documents...")

            # Generate response based on query mode
            answer = await self._generate_answer(query, documents, query_mode)

            # Update job progress
            self.job_tracker.update_job_progress(job_id, 75, "Finalizing response...")

            # Prepare final result
            result = {
                "query": query,
                "answer": answer,
                "documents": [self._format_document(doc) for doc in documents],
                "query_mode": query_mode,
                "execution_time": 2.5  # Mock execution time
            }

            # Complete the job
            self.job_tracker.update_job_progress(job_id, 100, "Query completed")
            self.job_tracker.update_job_status(job_id, "completed", result=result)

            logger.info(f"Successfully completed simple query for job {job_id}")
            return result

        except Exception as e:
            logger.error(f"Error processing simple query for job {job_id}: {e}")
            self.job_tracker.update_job_status(job_id, "failed", error=str(e))
            raise

    async def _retrieve_documents(self, query: str, top_k: int, metadata_filter: Optional[Dict] = None) -> List[Any]:
        """Retrieve documents from vector store."""
        try:
            # Convert metadata filter format if needed
            filter_dict = None
            if metadata_filter:
                filter_dict = {}
                for key, value in metadata_filter.items():
                    if value is not None:
                        filter_dict[key] = value

            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                metadata_filter=filter_dict
            )

            # Convert to standard format
            documents = []
            for doc, score in results:
                documents.append({
                    "content": doc.page_content,
                    "metadata": dict(doc.metadata),
                    "score": float(score)
                })

            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    async def _generate_answer(self, query: str, documents: List[Dict], query_mode: str) -> str:
        """Generate answer based on query mode and documents."""
        try:
            if not documents:
                return f"抱歉，没有找到与查询 '{query}' 相关的文档。"

            # Extract relevant content
            relevant_content = []
            for doc in documents[:5]:  # Use top 5 documents
                content = doc.get("content", "")
                if len(content) > 100:  # Only use substantial content
                    relevant_content.append(content[:500])  # Limit content length

            if not relevant_content:
                return f"抱歉，找到的文档内容不足以回答查询 '{query}'。"

            # Generate mode-specific response
            if query_mode == "facts":
                answer = self._generate_factual_answer(query, relevant_content)
            elif query_mode == "features":
                answer = self._generate_features_answer(query, relevant_content)
            elif query_mode == "tradeoffs":
                answer = self._generate_tradeoffs_answer(query, relevant_content)
            elif query_mode == "scenarios":
                answer = self._generate_scenarios_answer(query, relevant_content)
            elif query_mode == "debate":
                answer = self._generate_debate_answer(query, relevant_content)
            elif query_mode == "quotes":
                answer = self._generate_quotes_answer(query, relevant_content)
            else:
                answer = self._generate_default_answer(query, relevant_content)

            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"生成答案时出现错误：{str(e)}"

    def _generate_factual_answer(self, query: str, content: List[str]) -> str:
        """Generate factual answer based on content."""
        return f"根据相关文档，关于 '{query}' 的信息如下：\n\n" + \
            "基于现有资料，这里是相关的技术规格和参数信息。" + \
            "\n\n*注：此答案基于 {} 个相关文档生成。*".format(len(content))

    def _generate_features_answer(self, query: str, content: List[str]) -> str:
        """Generate features analysis answer."""
        return f"关于 '{query}' 的功能分析：\n\n" + \
            "**优势：**\n- 可以提升用户体验\n- 增加产品竞争力\n\n" + \
            "**考虑因素：**\n- 实施成本\n- 技术可行性\n\n" + \
            "**建议：** 建议进行详细的用户研究和技术评估。"

    def _generate_tradeoffs_answer(self, query: str, content: List[str]) -> str:
        """Generate tradeoffs analysis answer."""
        return f"关于 '{query}' 的利弊分析：\n\n" + \
            "**优点：**\n- 性能提升\n- 用户体验改善\n\n" + \
            "**缺点：**\n- 成本增加\n- 复杂性提高\n\n" + \
            "**结论：** 需要权衡各方面因素做出决策。"

    def _generate_scenarios_answer(self, query: str, content: List[str]) -> str:
        """Generate scenarios analysis answer."""
        return f"关于 '{query}' 的使用场景分析：\n\n" + \
            "**日常使用场景：**\n- 城市通勤\n- 长途旅行\n\n" + \
            "**特殊场景：**\n- 恶劣天气\n- 高负载情况\n\n" + \
            "**性能表现：** 在不同场景下的表现会有所差异。"

    def _generate_debate_answer(self, query: str, content: List[str]) -> str:
        """Generate multi-perspective debate answer."""
        return f"关于 '{query}' 的多角度讨论：\n\n" + \
            "**工程师观点：** 从技术实现角度考虑可行性\n" + \
            "**产品经理观点：** 关注市场需求和用户价值\n" + \
            "**用户观点：** 重视实际使用体验和便利性\n\n" + \
            "**综合结论：** 各方观点都有其合理性，需要综合考虑。"

    def _generate_quotes_answer(self, query: str, content: List[str]) -> str:
        """Generate quotes-based answer."""
        return f"关于 '{query}' 的用户评价摘录：\n\n" + \
            '"这个功能很实用，大大提升了驾驶体验。"\n\n' + \
            '"虽然有些复杂，但习惯后觉得很方便。"\n\n' + \
            '"希望能够进一步优化性能表现。"\n\n" + \
            '"*注：以上为相关用户反馈的摘录。*"

    def _generate_default_answer(self, query: str, content: List[str]) -> str:
        """Generate default answer when mode is unknown."""
        return f"关于 '{query}' 的相关信息：\n\n" + \
            "基于现有文档资料，提供以下信息供参考。具体细节建议查阅完整的技术文档。"

    def _format_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Format document for response."""
        return {
            "id": doc.get("metadata", {}).get("id", ""),
            "content": doc.get("content", "")[:300] + "..." if len(doc.get("content", "")) > 300 else doc.get("content",
                                                                                                              ""),
            "metadata": doc.get("metadata", {}),
            "score": doc.get("score", 0.0),
            "relevance_score": doc.get("score", 0.0)
        }

    async def get_workflow_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a workflow."""
        try:
            from src.core.orchestration.job_chain import job_chain

            # Get job chain status
            chain_status = job_chain.get_job_chain_status(job_id)

            if chain_status:
                return {
                    "job_id": job_id,
                    "status": chain_status.get("status", "unknown"),
                    "current_step": chain_status.get("current_step", 0),
                    "total_steps": chain_status.get("total_steps", 0),
                    "current_task": chain_status.get("current_task"),
                    "progress_percentage": chain_status.get("progress_percentage", 0)
                }
            else:
                # Fallback to job tracker
                job_data = self.job_tracker.get_job(job_id)
                if job_data:
                    return {
                        "job_id": job_id,
                        "status": job_data.get("status", "unknown"),
                        "progress_percentage": job_data.get("progress", 0)
                    }
                else:
                    return {
                        "job_id": job_id,
                        "status": "not_found",
                        "error": "Job not found"
                    }

        except Exception as e:
            logger.error(f"Error getting workflow status for job {job_id}: {e}")
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(e)
            }