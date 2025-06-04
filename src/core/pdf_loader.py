import os
import re
import tempfile
from typing import Dict, List, Optional, Union

import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models.schema import DocumentMetadata, DocumentSource


class PDFLoader:
    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            device: Optional[str] = None,
            use_ocr: bool = True,
            ocr_languages: str = "en+ch_doc"
    ):
        """
        Initialize the PDF loader.

        Args:
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            device: Device to use for GPU-accelerated OCR (cuda or cpu)
            use_ocr: Whether to use OCR for scanned PDFs
            ocr_languages: Languages to use for OCR (tesseract format)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_ocr = use_ocr
        self.ocr_languages = ocr_languages

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Initialize OCR if requested
        self.ocr_model = None
        if use_ocr:
            self._initialize_ocr()

    def _initialize_ocr(self):
        """Initialize OCR models with GPU support for both English and Chinese."""
        try:
            from paddleocr import PaddleOCR

            # Initialize PaddleOCR with GPU support and multilingual capability
            self.ocr_model = PaddleOCR(
                use_angle_cls=True,
                lang=self.ocr_languages,  # This will be "en+ch_doc"
                use_gpu=False,
                show_log=False
            )
            print(f"PaddleOCR initialized with GPU support: {self.device.startswith('cuda')}")
            print(f"OCR languages: {self.ocr_languages}")
        except ImportError:
            print("PaddleOCR not found. OCR functionality will be disabled.")
            print("To enable GPU-accelerated OCR, install with: pip install paddlepaddle-gpu paddleocr")
            self.use_ocr = False

    def load_pdf(self, file_path: str, use_ocr: Optional[bool] = None) -> List[Document]:
        """
        Load a PDF file using PyPDFLoader, with OCR fallback for scanned PDFs.

        Args:
            file_path: Path to the PDF file
            use_ocr: Override the instance's use_ocr setting

        Returns:
            List of Langchain Document objects (one per page)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # Use instance setting if not overridden
        use_ocr = self.use_ocr if use_ocr is None else use_ocr

        # Try standard PDF extraction first
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Check if we need OCR by examining if text was extracted
        total_text = sum(len(doc.page_content.strip()) for doc in documents)

        # If minimal text was extracted and OCR is enabled, apply OCR
        if total_text < 100 and use_ocr and self.ocr_model:
            print(f"PDF appears to be scanned or has minimal text. Applying OCR...")
            return self._apply_ocr(file_path, documents)

        return documents

    def _apply_ocr(self, file_path: str, original_documents: List[Document]) -> List[Document]:
        """
        Apply OCR to a PDF file with comprehensive Unicode handling.

        Args:
            file_path: Path to the PDF file
            original_documents: Original documents with minimal text

        Returns:
            List of documents with OCR-extracted text
        """
        try:
            import fitz  # PyMuPDF
            import numpy as np
            from PIL import Image
            from src.utils.unicode_handler import decode_unicode_escapes

            ocr_documents = []

            # Open the PDF
            pdf = fitz.open(file_path)

            for i, page in enumerate(pdf):
                # Get the page as a pixmap (image)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better OCR

                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Convert to numpy array for OCR
                img_np = np.array(img)

                # Apply OCR
                result = self.ocr_model.ocr(img_np, cls=True)

                # ENHANCED: Extract and clean OCR text with Unicode handling
                text = ""
                if result[0]:
                    for line in result[0]:
                        line_text = line[1][0]

                        # CRITICAL FIX: Apply Unicode decoding to OCR output
                        if isinstance(line_text, str):
                            # PaddleOCR sometimes returns Unicode escapes
                            line_text = decode_unicode_escapes(line_text)

                            # Additional OCR-specific cleaning
                            line_text = self._clean_ocr_text(line_text)

                        text += line_text + "\n"

                # VALIDATION: Check for remaining Unicode escapes
                if "\\u" in text:
                    print(f"Warning: Unicode escapes still present in OCR text for page {i + 1}")
                    # Apply additional decoding attempt
                    text = decode_unicode_escapes(text)

                # Create document with cleaned text
                ocr_doc = Document(
                    page_content=text,
                    metadata={
                        **original_documents[i].metadata,
                        "ocr_applied": True,
                        "page_number": i + 1,
                        "ocr_unicode_cleaned": True  # Flag for tracking
                    }
                )

                ocr_documents.append(ocr_doc)

            return ocr_documents

        except Exception as e:
            print(f"OCR with Unicode handling failed: {str(e)}. Falling back to original documents.")
            return original_documents

    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR-specific artifacts and encoding issues"""
        if not text:
            return text

        # Common OCR artifacts with Chinese text
        ocr_fixes = {
            # Common OCR misreads for Chinese characters
            # Add specific fixes as needed based on your OCR results
        }

        # Apply OCR fixes
        for old, new in ocr_fixes.items():
            text = text.replace(old, new)

        # Remove excessive whitespace that OCR sometimes introduces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def extract_automotive_metadata(self, text: str) -> Dict[str, any]:
        """
        Extract automotive-specific metadata from PDF text with Unicode handling.

        Args:
            text: Extracted text from PDF

        Returns:
            Dictionary with automotive metadata
        """
        from src.utils.unicode_handler import decode_unicode_escapes

        # CRITICAL FIX: Decode Unicode escapes before metadata extraction
        if isinstance(text, str) and "\\u" in text:
            print("Decoding Unicode escapes in PDF text before metadata extraction")
            text = decode_unicode_escapes(text)

        auto_metadata = {}
        text_lower = text.lower()

        # ENHANCED: Chinese manufacturer patterns with proper Unicode
        chinese_manufacturer_patterns = [
            (r"宝马|bmw", "宝马"),
            (r"奔驰|mercedes|mercedes-benz", "奔驰"),
            (r"奥迪|audi", "奥迪"),
            (r"丰田|toyota", "丰田"),
            (r"本田|honda", "本田"),
            (r"大众|volkswagen|vw", "大众"),
            (r"福特|ford", "福特"),
            (r"雪佛兰|chevrolet|chevy", "雪佛兰"),
            (r"日产|nissan", "日产"),
            (r"现代|hyundai", "现代"),
            (r"起亚|kia", "起亚"),
            (r"斯巴鲁|subaru", "斯巴鲁"),
            (r"马自达|mazda", "马自达"),
            (r"特斯拉|tesla", "特斯拉"),
            (r"沃尔沃|volvo", "沃尔沃"),
            (r"捷豹|jaguar", "捷豹"),
            (r"路虎|land rover", "路虎"),
            (r"雷克萨斯|lexus", "雷克萨斯"),
            (r"讴歌|acura", "讴歌"),
            (r"英菲尼迪|infiniti", "英菲尼迪"),
            (r"凯迪拉克|cadillac", "凯迪拉克"),
            (r"吉普|jeep", "吉普"),
        ]

        # Look for Chinese manufacturers first, then English
        for pattern, manufacturer in chinese_manufacturer_patterns:
            if re.search(pattern, text_lower):
                auto_metadata["manufacturer"] =