import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DocumentProcessingService:
    """Service for processing and cleaning documents."""

    def __init__(self):
        self.min_content_length = 10
        self.max_content_length = 5000
        self.max_documents_per_source = 3

    async def clean_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and process retrieved documents."""
        if not documents:
            logger.info("No documents to clean")
            return []

        logger.info(f"Cleaning {len(documents)} documents")
        cleaned = []

        for i, doc in enumerate(documents):
            try:
                cleaned_doc = self._clean_single_document(doc, i)
                if cleaned_doc and self._is_valid_document(cleaned_doc):
                    cleaned.append(cleaned_doc)
            except Exception as e:
                logger.error(f"Error cleaning document {i}: {e}")
                continue

        # Apply diversity filtering
        cleaned = self._apply_diversity_filtering(cleaned)

        logger.info(f"Cleaned {len(cleaned)} documents from {len(documents)} original documents")
        return cleaned

    def _clean_single_document(self, doc: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Clean a single document."""
        # Handle both Document objects and dictionaries
        if hasattr(doc, 'page_content'):
            # Langchain Document object
            content = doc.page_content
            metadata = dict(doc.metadata) if doc.metadata else {}
            score = getattr(doc, 'score', 0.0)
        elif isinstance(doc, tuple) and len(doc) == 2:
            # (Document, score) tuple
            document, score = doc
            content = document.page_content
            metadata = dict(document.metadata) if document.metadata else {}
        else:
            # Dictionary format
            content = doc.get("content", doc.get("page_content", ""))
            metadata = doc.get("metadata", {})
            score = doc.get("score", 0.0)

        # Clean content
        content = self._clean_content(content)

        # Ensure document has an ID
        doc_id = metadata.get("id", f"doc_{index}")
        if not doc_id:
            doc_id = f"doc_{index}"

        return {
            "id": doc_id,
            "content": content,
            "metadata": metadata,
            "score": float(score),
            "content_length": len(content),
            "relevance_score": float(score)
        }

    def _clean_content(self, content: str) -> str:
        """Clean document content."""
        if not content:
            return ""

        # Remove excessive whitespace
        content = " ".join(content.split())

        # Remove common artifacts
        content = content.replace("\n\n\n", "\n\n")
        content = content.replace("  ", " ")

        # Trim to max length
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."

        return content.strip()

    def _is_valid_document(self, doc: Dict[str, Any]) -> bool:
        """Check if document meets quality criteria."""
        content = doc.get("content", "")

        # Check minimum content length
        if len(content) < self.min_content_length:
            return False

        # Check for meaningful content (not just punctuation/numbers)
        meaningful_chars = sum(1 for c in content if c.isalpha())
        if meaningful_chars < 5:
            return False

        # Check score threshold
        score = doc.get("score", 0.0)
        if score < 0.1:  # Very low relevance
            return False

        return True

    def _apply_diversity_filtering(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply diversity filtering to avoid too many documents from same source."""
        if not documents:
            return documents

        # Group by source
        source_groups = {}
        for doc in documents:
            source = doc.get("metadata", {}).get("source", "unknown")
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(doc)

        # Limit documents per source
        filtered_docs = []
        for source, docs in source_groups.items():
            # Sort by score (highest first)
            docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            # Take top N documents from this source
            filtered_docs.extend(docs[:self.max_documents_per_source])

        # Sort final list by score
        filtered_docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        return filtered_docs

    def calculate_processing_statistics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate processing statistics for documents."""
        if not documents:
            return {
                "total_documents": 0,
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "avg_content_length": 0,
                "sources_by_type": {},
                "score_distribution": {
                    "high_relevance": 0,
                    "medium_relevance": 0,
                    "low_relevance": 0
                },
                "content_quality": {
                    "substantial_content": 0,
                    "medium_content": 0,
                    "minimal_content": 0
                }
            }

        scores = [doc.get("score", 0) for doc in documents]
        content_lengths = [len(doc.get("content", "")) for doc in documents]

        # Count sources by type
        sources_by_type = {}
        for doc in documents:
            metadata = doc.get("metadata", {})
            source_type = metadata.get("source", "unknown")
            sources_by_type[source_type] = sources_by_type.get(source_type, 0) + 1

        # Score distribution
        high_relevance = len([s for s in scores if s > 0.8])
        medium_relevance = len([s for s in scores if 0.5 <= s <= 0.8])
        low_relevance = len([s for s in scores if s < 0.5])

        # Content quality distribution
        substantial_content = len([l for l in content_lengths if l > 500])
        medium_content = len([l for l in content_lengths if 100 <= l <= 500])
        minimal_content = len([l for l in content_lengths if l < 100])

        return {
            "total_documents": len(documents),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "avg_content_length": sum(content_lengths) / len(content_lengths),
            "sources_by_type": sources_by_type,
            "score_distribution": {
                "high_relevance": high_relevance,
                "medium_relevance": medium_relevance,
                "low_relevance": low_relevance
            },
            "content_quality": {
                "substantial_content": substantial_content,
                "medium_content": medium_content,
                "minimal_content": minimal_content
            }
        }

    def extract_key_information(self, documents: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Extract key information from documents relevant to the query."""
        if not documents:
            return {"key_points": [], "summary": "No documents available for analysis."}

        # Simple keyword extraction based on query
        query_keywords = query.lower().split()

        key_points = []
        relevant_sentences = []

        for doc in documents[:5]:  # Analyze top 5 documents
            content = doc.get("content", "")
            sentences = content.split("。")  # Split by Chinese period

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:  # Skip very short sentences
                    continue

                # Check if sentence contains query keywords
                sentence_lower = sentence.lower()
                keyword_matches = sum(1 for keyword in query_keywords if keyword in sentence_lower)

                if keyword_matches > 0:
                    relevance_score = keyword_matches / len(query_keywords)
                    relevant_sentences.append({
                        "sentence": sentence,
                        "relevance": relevance_score,
                        "source": doc.get("metadata", {}).get("source", "unknown")
                    })

        # Sort by relevance and take top sentences
        relevant_sentences.sort(key=lambda x: x["relevance"], reverse=True)

        # Extract key points
        for item in relevant_sentences[:10]:  # Top 10 relevant sentences
            if len(item["sentence"]) > 20:  # Only substantial sentences
                key_points.append({
                    "text": item["sentence"][:200] + ("..." if len(item["sentence"]) > 200 else ""),
                    "relevance": item["relevance"],
                    "source": item["source"]
                })

        # Generate summary
        if key_points:
            summary = f"基于 {len(documents)} 个文档，提取到 {len(key_points)} 个关键信息点。"
        else:
            summary = f"分析了 {len(documents)} 个文档，但未找到与查询直接相关的关键信息。"

        return {
            "key_points": key_points,
            "summary": summary,
            "total_sentences_analyzed": len(relevant_sentences),
            "documents_analyzed": min(len(documents), 5)
        }

    def validate_document_quality(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the quality of processed documents."""
        if not documents:
            return {
                "quality_score": 0.0,
                "issues": ["No documents to validate"],
                "recommendations": ["Ensure documents are retrieved before validation"]
            }

        issues = []
        recommendations = []
        quality_scores = []

        for i, doc in enumerate(documents):
            doc_score = 0.0

            # Check content length
            content_length = len(doc.get("content", ""))
            if content_length < 50:
                issues.append(f"Document {i}: Very short content ({content_length} chars)")
                doc_score += 0.2
            elif content_length < 200:
                doc_score += 0.6
            else:
                doc_score += 1.0

            # Check relevance score
            relevance = doc.get("score", 0.0)
            if relevance < 0.3:
                issues.append(f"Document {i}: Low relevance score ({relevance:.2f})")
                doc_score += 0.2
            elif relevance < 0.6:
                doc_score += 0.6
            else:
                doc_score += 1.0

            # Check metadata completeness
            metadata = doc.get("metadata", {})
            if not metadata.get("source"):
                issues.append(f"Document {i}: Missing source information")
                doc_score += 0.0
            else:
                doc_score += 0.5

            quality_scores.append(doc_score / 2.5)  # Normalize to 0-1

        overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        # Generate recommendations
        if overall_quality < 0.5:
            recommendations.append("Consider adjusting search parameters to get more relevant documents")
            recommendations.append("Check if the query is too specific or if more data is needed")
        elif overall_quality < 0.7:
            recommendations.append("Document quality is acceptable but could be improved")
            recommendations.append("Consider using a higher top_k value for more document options")
        else:
            recommendations.append("Document quality is good")

        return {
            "quality_score": overall_quality,
            "individual_scores": quality_scores,
            "issues": issues,
            "recommendations": recommendations,
            "documents_validated": len(documents)
        }