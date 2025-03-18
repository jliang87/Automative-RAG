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
    """
    Enhanced class for loading and processing PDF documents with GPU acceleration.
    
    Extracts text and metadata from PDFs, with special handling
    for automotive service manuals and specification sheets.
    Adds OCR capabilities for scanned PDFs using GPU acceleration.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        device: Optional[str] = None,
        use_ocr: bool = True,
        ocr_languages: str = "en"
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
        """Initialize OCR models with GPU support if available."""
        try:
            from paddleocr import PaddleOCR
            
            # Initialize PaddleOCR with GPU support
            self.ocr_model = PaddleOCR(
                use_angle_cls=True,
                lang=self.ocr_languages,
                use_gpu=self.device.startswith("cuda"),
                show_log=False
            )
            print(f"PaddleOCR initialized with GPU support: {self.device.startswith('cuda')}")
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
        Apply OCR to a PDF file using GPU acceleration.
        
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
                
                # Extract text from OCR result
                text = ""
                if result[0]:
                    for line in result[0]:
                        text += line[1][0] + "\n"
                
                # Create a new document with the OCR text
                ocr_doc = Document(
                    page_content=text,
                    metadata={
                        **original_documents[i].metadata,
                        "ocr_applied": True,
                        "page_number": i + 1
                    }
                )
                
                ocr_documents.append(ocr_doc)
            
            return ocr_documents
        except Exception as e:
            print(f"OCR failed: {str(e)}. Falling back to original documents.")
            return original_documents

    def extract_automotive_metadata(self, text: str) -> Dict[str, any]:
        """
        Extract automotive-specific metadata from PDF text.

        Args:
            text: Extracted text from PDF

        Returns:
            Dictionary with automotive metadata
        """
        auto_metadata = {}
        
        # Common manufacturers with regex patterns
        manufacturer_patterns = [
            (r"toyota", "Toyota"),
            (r"honda", "Honda"),
            (r"ford", "Ford"),
            (r"chevrolet|chevy", "Chevrolet"),
            (r"bmw", "BMW"),
            (r"mercedes|mercedes-benz", "Mercedes-Benz"),
            (r"audi", "Audi"),
            (r"volkswagen|vw", "Volkswagen"),
            (r"nissan", "Nissan"),
            (r"hyundai", "Hyundai"),
            (r"kia", "Kia"),
            (r"subaru", "Subaru"),
            (r"mazda", "Mazda"),
            (r"porsche", "Porsche"),
            (r"ferrari", "Ferrari"),
            (r"lamborghini", "Lamborghini"),
            (r"tesla", "Tesla"),
            (r"volvo", "Volvo"),
            (r"jaguar", "Jaguar"),
            (r"land rover", "Land Rover"),
            (r"lexus", "Lexus"),
            (r"acura", "Acura"),
            (r"infiniti", "Infiniti"),
            (r"cadillac", "Cadillac"),
            (r"jeep", "Jeep"),
        ]
        
        # Look for manufacturer
        text_lower = text.lower()
        for pattern, manufacturer in manufacturer_patterns:
            if re.search(pattern, text_lower):
                auto_metadata["manufacturer"] = manufacturer
                break
                
        # Extract year (4-digit number between 1900 and 2100)
        year_match = re.search(r'(19\d{2}|20\d{2})', text)
        if year_match:
            auto_metadata["year"] = int(year_match.group(0))
            
        # Try to extract model
        # This is more complex and varies by manufacturer
        # Look for common patterns after manufacturer name
        if "manufacturer" in auto_metadata:
            manufacturer = auto_metadata["manufacturer"]
            
            # Pattern: "<Manufacturer> <Model>"
            model_pattern = rf"{manufacturer}\s+([A-Z0-9][-A-Za-z0-9\s]+?)[\s\.,]"
            model_match = re.search(model_pattern, text)
            
            if model_match:
                auto_metadata["model"] = model_match.group(1).strip()
                
        # Categories
        category_patterns = [
            (r"sedan", "sedan"),
            (r"suv|crossover", "suv"),
            (r"truck|pickup", "truck"),
            (r"sports car|supercar|hypercar", "sports"),
            (r"minivan|van", "minivan"),
            (r"coup[eé]", "coupe"),
            (r"convertible|cabriolet", "convertible"),
            (r"hatchback", "hatchback"),
            (r"wagon|estate", "wagon"),
        ]
        
        for pattern, category in category_patterns:
            if re.search(pattern, text_lower):
                auto_metadata["category"] = category
                break
                
        # Engine types
        engine_patterns = [
            (r"gasoline|petrol|gas engine", "gasoline"),
            (r"diesel", "diesel"),
            (r"electric|ev|battery-powered", "electric"),
            (r"hybrid|phev|plug-in", "hybrid"),
            (r"hydrogen|fuel cell", "hydrogen"),
        ]
        
        for pattern, engine_type in engine_patterns:
            if re.search(pattern, text_lower):
                auto_metadata["engine_type"] = engine_type
                break
                
        # Transmission types
        transmission_patterns = [
            (r"automatic transmission|auto transmission", "automatic"),
            (r"manual transmission|stick shift", "manual"),
            (r"cvt|continuously variable", "cvt"),
            (r"dct|dual-clutch", "dct"),
        ]
        
        for pattern, transmission in transmission_patterns:
            if re.search(pattern, text_lower):
                auto_metadata["transmission"] = transmission
                break
        
        return auto_metadata

    def extract_tables(self, file_path: str) -> List[Dict]:
        """
        Extract tables from PDF using GPU-accelerated detection.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing table data
        """
        try:
            import camelot
            import pandas as pd
            
            # Extract tables
            tables = camelot.read_pdf(
                file_path,
                pages='all',
                flavor='lattice'  # Try both 'lattice' and 'stream' for different table types
            )
            
            result = []
            
            # Convert tables to structured data
            for i, table in enumerate(tables):
                # Convert to pandas DataFrame
                df = table.df
                
                # Convert DataFrame to dictionary
                table_dict = {
                    "table_id": i + 1,
                    "page_number": table.page,
                    "data": df.to_dict(orient='records'),
                    "headers": df.columns.tolist()
                }
                
                result.append(table_dict)
                
            return result
        except ImportError:
            print("Camelot-py not installed. Table extraction disabled.")
            print("To enable table extraction, install: pip install camelot-py opencv-python ghostscript")
            return []
        except Exception as e:
            print(f"Table extraction failed: {str(e)}")
            return []

    def process_pdf(
        self,
        file_path: str,
        custom_metadata: Optional[Dict[str, str]] = None,
        extract_tables: bool = True
    ) -> List[Document]:
        """
        Process a PDF file and return chunked Langchain documents with GPU-accelerated features.

        Args:
            file_path: Path to the PDF file
            custom_metadata: Optional custom metadata
            extract_tables: Whether to extract tables from the PDF

        Returns:
            List of Langchain Document objects with metadata
        """
        # Load PDF with OCR if needed
        raw_documents = self.load_pdf(file_path)
        
        # Combine all pages to extract metadata
        full_text = " ".join([doc.page_content for doc in raw_documents])
        
        # Extract automotive metadata
        auto_metadata = self.extract_automotive_metadata(full_text)
        
        # Extract tables if requested
        tables = []
        if extract_tables:
            try:
                tables = self.extract_tables(file_path)
            except Exception as e:
                print(f"Table extraction error: {str(e)}")
        
        # Create metadata object
        base_metadata = DocumentMetadata(
            source=DocumentSource.PDF,
            source_id=os.path.basename(file_path),
            url=None,
            title=custom_metadata.get("title") if custom_metadata else os.path.basename(file_path),
            author=custom_metadata.get("author"),
            published_date=None,
            manufacturer=auto_metadata.get("manufacturer"),
            model=auto_metadata.get("model"),
            year=auto_metadata.get("year"),
            category=auto_metadata.get("category"),
            engine_type=auto_metadata.get("engine_type"),
            transmission=auto_metadata.get("transmission"),
            custom_metadata=custom_metadata or {},
        )
        
        # Add extracted tables to metadata if any were found
        if tables:
            base_metadata.custom_metadata["has_tables"] = True
            base_metadata.custom_metadata["table_count"] = len(tables)
            
            # Store tables in a simplified format in metadata
            # (full table data would be too large for metadata)
            table_info = []
            for table in tables:
                table_info.append({
                    "table_id": table["table_id"],
                    "page_number": table["page_number"],
                    "columns": len(table["headers"]),
                    "rows": len(table["data"])
                })
            base_metadata.custom_metadata["table_info"] = table_info
        
        # Update document metadata
        for doc in raw_documents:
            doc.metadata.update(base_metadata.dict())
            
            # Add OCR info if present
            if "ocr_applied" in doc.metadata:
                doc.metadata["custom_metadata"]["ocr_applied"] = True
        
        # Split documents into chunks
        chunked_documents = self.text_splitter.split_documents(raw_documents)
        
        return chunked_documents