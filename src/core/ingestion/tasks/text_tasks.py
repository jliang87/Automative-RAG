"""
Text ingestion tasks - Extracted from JobChain
Handles text processing workflows
"""

import time
import logging
from typing import Dict, Optional
import dramatiq

from src.core.orchestration.job_tracker import job_tracker
from src.core.orchestration.job_chain import job_chain
from src.core.orchestration.queue_manager import queue_manager, QueueNames

logger = logging.getLogger(__name__)

# Import unified ingestion system (internal use only)
try:
    from src.core.ingestion.factory import ProcessorFactory

    ENHANCED_PROCESSING_AVAILABLE = True
    logger.info("✅ Enhanced ingestion system available for text processing")
except ImportError:
    ENHANCED_PROCESSING_AVAILABLE = False
    logger.warning("⚠️ Enhanced ingestion system not available, falling back to basic text processing")


@queue_manager.create_task_decorator(QueueNames.CPU_TASKS.value)
def process_text_task(job_id: str, text: str, metadata: Optional[Dict] = None):
    """Process text - NOW uses Enhanced Ingestion System for superior results!"""
    try:
        logger.info(f"Processing text for job {job_id}")

        if ENHANCED_PROCESSING_AVAILABLE:
            # ✅ NEW: Use enhanced text processor with EnhancedTranscriptProcessor
            logger.info(f"🔧 Using enhanced text processor for job {job_id}")

            processor = ProcessorFactory.create_processor("text")
            documents = processor.process(text, metadata)

            # Convert documents to format for next task
            document_dicts = []
            for doc in documents:
                document_dicts.append({
                    "content": doc.page_content,  # ✅ NOW contains embedded metadata!
                    "metadata": doc.metadata  # ✅ NOW contains automotive metadata!
                })

            logger.info(f"✅ Enhanced text processing completed for job {job_id}: {len(documents)} documents")

            # Log enhancement results
            if documents:
                sample_doc = documents[0]
                vehicle_detected = sample_doc.metadata.get('vehicleDetected', False)
                metadata_injected = sample_doc.metadata.get('metadataInjected', False)

                logger.info(f"🚗 Vehicle detected in text: {vehicle_detected}")
                logger.info(f"🏷️ Metadata injected in text: {metadata_injected}")

                # Show embedded patterns
                import re
                embedded_patterns = re.findall(r'【[^】]+】', sample_doc.page_content)
                logger.info(f"📝 Text embedded patterns: {len(embedded_patterns)}")

            text_result = {
                "documents": document_dicts,
                "chunk_count": len(documents),
                "text_processing_completed_at": time.time(),
                "original_text": text,
                "custom_metadata": metadata,
                # ✅ NEW: Enhanced processing markers
                "enhanced_processing_used": True,
                "metadata_injection_applied": True,
                "processing_method": "enhanced_transcript_processor",
                "unified_ingestion_system": True,
                "automotive_metadata_extracted": any(
                    doc.metadata.get('vehicleDetected', False) for doc in documents
                )
            }

        else:
            # Fallback to basic text processing (original logic)
            logger.warning(f"Enhanced processing not available, using basic text processing for job {job_id}")

            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_core.documents import Document
            from src.config.settings import settings

            # Validate text input
            if not text or not text.strip():
                error_msg = f"Text input is empty for job {job_id}"
                logger.error(error_msg)
                job_chain.task_failed(job_id, error_msg)
                return

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

            chunks = text_splitter.split_text(text)

            # Create documents
            documents = []
            for i, chunk_text in enumerate(chunks):
                # Apply metadata extraction to the full text (not just chunks)
                if i == 0:  # Only extract metadata once from full text
                    try:
                        from src.utils.helpers import extract_metadata_from_text
                        extracted_metadata = extract_metadata_from_text(text)
                    except ImportError:
                        extracted_metadata = {}
                else:
                    extracted_metadata = {}

                # Combine extracted metadata with provided metadata
                doc_metadata = {
                    "chunk_id": i,
                    "source": "manual",
                    "source_id": job_id,
                    "total_chunks": len(chunks),
                    **extracted_metadata,
                    **(metadata or {})
                }

                doc = Document(
                    page_content=chunk_text,
                    metadata=doc_metadata
                )
                documents.append(doc)

            logger.info(f"Text processing completed for job {job_id}: {len(chunks)} chunks")

            # Convert documents to format for next task
            document_dicts = []
            for doc in documents:
                document_dicts.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

            text_result = {
                "documents": document_dicts,
                "chunk_count": len(chunks),
                "text_processing_completed_at": time.time(),
                "original_text": text,
                "custom_metadata": metadata,
                "enhanced_processing_used": False,
                "processing_method": "basic_text_splitting"
            }

        # Store result in job tracker
        job_tracker.update_job_status(
            job_id,
            "processing",
            result=text_result,
            stage="text_processing_completed",
            replace_result=True
        )

        # Trigger next task
        job_chain.task_completed(job_id, text_result)

    except Exception as e:
        logger.error(f"Text processing failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"Text processing failed: {str(e)}")


def start_text_processing(job_id: str, data: Dict):
    """
    Start text processing workflow

    Args:
        job_id: Job identifier
        data: Job data containing text and metadata
    """
    logger.info(f"Starting text processing workflow for job {job_id}")

    # Validate required data
    if "text" not in data:
        error_msg = "text required for text processing"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)
        return

    # Start the text processing task
    process_text_task.send(job_id, data["text"], data.get("metadata"))