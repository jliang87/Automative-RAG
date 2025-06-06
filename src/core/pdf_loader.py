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
                lang=self.ocr_languages,
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
        try:
            import fitz  # PyMuPDF
            import numpy as np
            from PIL import Image

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

                # Extract OCR text - no manual Unicode cleaning needed
                text = ""
                if result[0]:
                    for line in result[0]:
                        line_text = line[1][0]

                        # Basic OCR cleaning only
                        if isinstance(line_text, str):
                            line_text = self._clean_ocr_text(line_text)

                        text += line_text + "\n"

                # Create document with OCR text
                ocr_doc = Document(
                    page_content=text,
                    metadata={
                        **original_documents[i].metadata,
                        "ocr_applied": True,
                        "page_number": i + 1,
                    }
                )

                ocr_documents.append(ocr_doc)

            return ocr_documents

        except Exception as e:
            print(f"OCR failed: {str(e)}. Falling back to original documents.")
            return original_documents

    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR-specific artifacts (not Unicode - that's handled by global patch)"""
        if not text:
            return text

        # Remove excessive whitespace that OCR sometimes introduces
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def extract_automotive_metadata(self, text: str) -> Dict[str, any]:
        auto_metadata = {}
        text_lower = text.lower()

        # Chinese manufacturer patterns
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
                auto_metadata["manufacturer"] = manufacturer
                break

        # Chinese model patterns
        chinese_model_patterns = [
            # BMW models
            (r"[1-8]系|X[1-7]|i[3-8]|Z4", "宝马"),
            # Mercedes models
            (r"[A-Z]级|GLA|GLC|GLE|GLS|AMG", "奔驰"),
            # Audi models
            (r"A[1-8]|Q[2-8]|TT|R8", "奥迪"),
            # Toyota models
            (r"凯美瑞|卡罗拉|汉兰达|普拉多|陆地巡洋舰|camry|corolla|highlander|prado|land cruiser", "丰田"),
            # Honda models
            (r"雅阁|思域|crv|奥德赛|accord|civic|odyssey", "本田"),
        ]

        # Extract model if manufacturer is known
        if "manufacturer" in auto_metadata:
            manufacturer = auto_metadata["manufacturer"]
            for pattern, manu in chinese_model_patterns:
                if manufacturer == manu and re.search(pattern, text_lower, re.IGNORECASE):
                    match = re.search(pattern, text_lower, re.IGNORECASE)
                    if match:
                        auto_metadata["model"] = match.group(0)
                        break

        # Chinese category detection
        chinese_categories = [
            (r"轿车|sedan|saloon", "轿车"),
            (r"suv|越野车|运动型多用途车|sport utility vehicle", "SUV"),
            (r"truck|pickup|卡车|皮卡|货车", "卡车"),
            (r"sports car|supercar|跑车|运动车", "跑车"),
            (r"minivan|van|面包车|mpv|多功能车", "面包车"),
            (r"coupe|coupé|双门轿跑|轿跑车", "轿跑"),
            (r"convertible|cabriolet|敞篷车|软顶车", "敞篷车"),
            (r"hatchback|hot hatch|掀背车|两厢车", "掀背车"),
            (r"wagon|estate|旅行车|瓦罐车", "旅行车"),
        ]

        for pattern, category in chinese_categories:
            if re.search(pattern, text_lower):
                auto_metadata["category"] = category
                break

        # Chinese engine type detection
        chinese_engine_types = [
            (r"汽油|gasoline|petrol|gas engine|汽油机", "汽油"),
            (r"柴油|diesel|柴油机", "柴油"),
            (r"电动|electric|ev|纯电|电池|battery|pure electric", "电动"),
            (r"混合动力|hybrid|油电混合|插电混合|phev|plug-in hybrid", "混合动力"),
            (r"氢燃料|hydrogen|fuel cell|氢气|燃料电池", "氢燃料"),
        ]

        for pattern, engine_type in chinese_engine_types:
            if re.search(pattern, text_lower):
                auto_metadata["engine_type"] = engine_type
                break

        # Chinese transmission detection
        chinese_transmissions = [
            (r"自动|automatic|auto|自动挡|自动变速箱", "自动"),
            (r"手动|manual|stick|手动挡|手动变速箱|manual transmission", "手动"),
            (r"cvt|无级变速|continuously variable|无级变速箱", "CVT"),
            (r"dct|dual-clutch|双离合|双离合变速箱", "双离合"),
        ]

        for pattern, transmission in chinese_transmissions:
            if re.search(pattern, text_lower):
                auto_metadata["transmission"] = transmission
                break

        # Year extraction with validation
        year_patterns = [
            r"(\d{4})年",  # Chinese year format
            r"(20[0-9]{2})款",  # Model year format
            r"(19|20)\d{2}",  # Standard year format
        ]

        for pattern in year_patterns:
            match = re.search(pattern, text)
            if match:
                year = int(match.group(1)) if match.group(1).isdigit() else int(match.group(0))
                if 1990 <= year <= 2030:  # Reasonable automotive year range
                    auto_metadata["year"] = year
                    break

        return auto_metadata

    def process_pdf(
            self,
            file_path: str,
            custom_metadata: Optional[Dict[str, str]] = None,
            extract_tables: bool = True
    ) -> List[Document]:
        # Load the PDF
        documents = self.load_pdf(file_path)

        # Process each document
        processed_documents = []

        for i, doc in enumerate(documents):
            # Content is already clean thanks to global patch
            content = doc.page_content

            # Extract automotive metadata
            auto_metadata = self.extract_automotive_metadata(content)

            # Create comprehensive metadata
            enhanced_metadata = {
                # Basic document info
                "source": "pdf",
                "source_id": os.path.basename(file_path),
                "page_number": i + 1,
                "total_pages": len(documents),
                "file_path": file_path,

                # Merge with existing metadata
                **doc.metadata,

                # Add automotive metadata
                **auto_metadata,

                # Add custom metadata if provided
                **(custom_metadata or {}),
            }

            # Create enhanced document
            enhanced_doc = Document(
                page_content=content,
                metadata=enhanced_metadata
            )

            processed_documents.append(enhanced_doc)

        # Split into chunks if documents are large
        final_documents = []
        for doc in processed_documents:
            if len(doc.page_content) > self.chunk_size:
                # Split large documents into chunks
                chunks = self.text_splitter.split_text(doc.page_content)

                for j, chunk in enumerate(chunks):
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": j,
                        "total_chunks": len(chunks),
                        "is_chunk": True
                    })

                    chunk_doc = Document(
                        page_content=chunk,
                        metadata=chunk_metadata
                    )
                    final_documents.append(chunk_doc)
            else:
                # Keep document as-is if it's not too large
                final_documents.append(doc)

        print(f"PDF processing completed: {len(final_documents)} documents")
        return final_documents

    def extract_tables(self, file_path: str) -> List[Dict[str, any]]:
        try:
            import camelot

            # Extract tables using camelot
            tables = camelot.read_pdf(file_path, pages='all')

            table_data = []
            for i, table in enumerate(tables):
                # Convert table to dictionary format
                df = table.df

                table_dict = {
                    "table_id": i,
                    "page": table.page,
                    "accuracy": table.accuracy,
                    "data": df.to_dict('records'),
                    "headers": list(df.columns),
                }

                table_data.append(table_dict)

            print(f"Extracted {len(table_data)} tables")
            return table_data

        except ImportError:
            print("Camelot not available for table extraction. Install with: pip install camelot-py[cv]")
            return []
        except Exception as e:
            print(f"Error extracting tables: {str(e)}")
            return []

    def get_pdf_info(self, file_path: str) -> Dict[str, any]:
        """
        Get PDF file information.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with PDF information
        """
        try:
            import fitz

            pdf = fitz.open(file_path)
            metadata = pdf.metadata

            pdf_info = {
                "file_path": file_path,
                "file_size": os.path.getsize(file_path),
                "page_count": pdf.page_count,
                "metadata": metadata,
            }

            pdf.close()
            return pdf_info

        except Exception as e:
            print(f"Error getting PDF info: {str(e)}")
            return {"file_path": file_path, "error": str(e)}