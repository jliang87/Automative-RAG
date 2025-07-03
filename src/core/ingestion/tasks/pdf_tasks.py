"""
PDF ingestion tasks - Extracted from JobChain
Handles PDF processing workflows
"""

import time
import logging
from typing import Dict, Optional
import dramatiq

from src.core.orchestration.job_tracker import job_tracker
from src.core.orchestration.job_chain import job_chain

logger = logging.getLogger(__name__)

# Import unified ingestion system (internal use only)
try:
    from core.ingestion.factory import ProcessorFactory

    ENHANCED_PROCESSING_AVAILABLE = True
    logger.info("✅ Enhanced ingestion system available for PDF processing")
except ImportError:
    ENHANCED_PROCESSING_AVAILABLE = False
    logger.warning("⚠️ Enhanced ingestion system not available, falling back to basic PDF processing")


@dramatiq.actor(queue_name="cpu_tasks", store_results=True, max_retries=2)
def process_pdf_task(job_id: str, file_path: str, metadata: Optional[Dict] = None):
    """Process PDF - NOW uses Enhanced Ingestion System for consistent metadata!"""
    try:
        logger.info(f"Processing PDF for job {job_id}: {file_path}")

        if ENHANCED_PROCESSING_AVAILABLE:
            # ✅ NEW: Use enhanced PDF processor with EnhancedTranscriptProcessor
            logger.info(f"🔧 Using enhanced PDF processor for job {job_id}")

            processor = ProcessorFactory.create_processor("pdf")
            documents = processor.process(file_path, metadata)

            # Convert documents to format for next task
            document_dicts = []
            for doc in documents:
                document_dicts.append({
                    "content": doc.page_content,  # ✅ NOW contains embedded metadata!
                    "metadata": doc.metadata  # ✅ NOW contains automotive metadata!
                })

            logger.info(f"✅ Enhanced PDF processing completed for job {job_id}: {len(documents)} documents")

            # Log enhancement results
            if documents:
                sample_doc = documents[0]
                vehicle_detected = sample_doc.metadata.get('vehicleDetected', False)
                metadata_injected = sample_doc.metadata.get('metadataInjected', False)

                logger.info(f"🚗 Vehicle detected in PDF: {vehicle_detected}")
                logger.info(f"🏷️ Metadata injected in PDF: {metadata_injected}")

                # Show embedded patterns
                import re
                embedded_patterns = re.findall(r'【[^】]+】', sample_doc.page_content)
                logger.info(f"📝 PDF embedded patterns: {len(embedded_patterns)}")

            pdf_result = {
                "documents": document_dicts,
                "document_count": len(documents),
                "pdf_processing_completed_at": time.time(),
                "file_path": file_path,
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
            # Fallback to basic PDF processing
            logger.warning(f"Enhanced processing not available, using basic PDF processing for job {job_id}")

            from src.core.ingestion.loaders.pdf_loader import PDFLoader
            from src.config.settings import settings

            # Create PDF loader
            pdf_loader = PDFLoader(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                device="cpu",
                use_ocr=getattr(settings, 'use_pdf_ocr', True),
                ocr_languages=getattr(settings, 'ocr_languages', 'en+ch_doc')
            )

            # Process PDF
            documents = pdf_loader.process_pdf(
                file_path=file_path,
                custom_metadata=metadata,
            )

            logger.info(f"PDF processing completed for job {job_id}: {len(documents)} documents")

            # Convert documents to format for next task
            document_dicts = []
            for doc in documents:
                document_dicts.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

            pdf_result = {
                "documents": document_dicts,
                "document_count": len(documents),
                "pdf_processing_completed_at": time.time(),
                "file_path": file_path,
                "custom_metadata": metadata,
                "enhanced_processing_used": False,
                "processing_method": "basic_pdf_loader"
            }

        # Store result in job tracker
        job_tracker.update_job_status(
            job_id,
            "processing",
            result=pdf_result,
            stage="pdf_processing_completed",
            replace_result=True
        )

        # Trigger next task
        job_chain.task_completed(job_id, pdf_result)

    except Exception as e:
        logger.error(f"PDF processing failed for job {job_id}: {str(e)}")
        job_chain.task_failed(job_id, f"PDF processing failed: {str(e)}")


def start_pdf_processing(job_id: str, data: Dict):
    """
    Start PDF processing workflow

    Args:
        job_id: Job identifier
        data: Job data containing file_path and metadata
    """
    logger.info(f"Starting PDF processing workflow for job {job_id}")

    # Validate required data
    if "file_path" not in data:
        error_msg = "file_path required for PDF processing"
        logger.error(error_msg)
        job_chain.task_failed(job_id, error_msg)
        return

    # Start the PDF processing task
    process_pdf_task.send(job_id, data["file_path"], data.get("metadata"))