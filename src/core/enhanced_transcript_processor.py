import re
import jieba
import jieba.analyse
from typing import Dict, List, Optional, Tuple, Any, Set
from langchain_core.documents import Document


class MetadataExtractor:
    """Extract and separate recognized vs unrecognized metadata fields."""

    def __init__(self):
        self._setup_extraction_patterns()

    def _setup_extraction_patterns(self):
        """Setup patterns for extracting recognizable metadata fields."""

        # Vehicle manufacturers (Chinese names for Chinese query support)
        self.manufacturers = {
            # Chinese brands (Chinese names)
            '吉利': '吉利',
            'geely': '吉利',
            '比亚迪': '比亚迪',
            'byd': '比亚迪',
            '长城': '长城',
            '哈弗': '长城',  # Haval is Great Wall's SUV brand
            '蔚来': '蔚来',
            'nio': '蔚来',
            '理想': '理想',
            '小鹏': '小鹏',
            'xpeng': '小鹏',
            '奇瑞': '奇瑞',
            '长安': '长安',

            # International brands (Chinese names)
            '宝马': '宝马',
            'bmw': '宝马',
            '奔驰': '奔驰',
            'mercedes': '奔驰',
            'benz': '奔驰',
            '奥迪': '奥迪',
            'audi': '奥迪',
            '特斯拉': '特斯拉',
            'tesla': '特斯拉',
            '丰田': '丰田',
            'toyota': '丰田',
            '本田': '本田',
            'honda': '本田',
            '大众': '大众',
            'volkswagen': '大众',
            'vw': '大众',
            '福特': '福特',
            'ford': '福特',
            '日产': '日产',
            'nissan': '日产',
            '现代': '现代',
            'hyundai': '现代',
            '起亚': '起亚',
            'kia': '起亚',
        }

        # Vehicle models (Chinese names with proper categorization)
        self.vehicle_models = {
            # Geely models (separate vehicleType and fuelType)
            '星越L': {'manufacturer': '吉利', 'model': '星越L', 'vehicleType': 'SUV', 'fuelType': '汽油'},
            '星越': {'manufacturer': '吉利', 'model': '星越', 'vehicleType': 'SUV', 'fuelType': '汽油'},
            '缤越': {'manufacturer': '吉利', 'model': '缤越', 'vehicleType': 'SUV', 'fuelType': '汽油'},
            '帝豪': {'manufacturer': '吉利', 'model': '帝豪', 'vehicleType': '轿车', 'fuelType': '汽油'},
            '博越': {'manufacturer': '吉利', 'model': '博越', 'vehicleType': 'SUV', 'fuelType': '汽油'},
            '几何A': {'manufacturer': '吉利', 'model': '几何A', 'vehicleType': '轿车', 'fuelType': '电动'},
            '几何C': {'manufacturer': '吉利', 'model': '几何C', 'vehicleType': 'SUV', 'fuelType': '电动'},

            # BYD models (separate vehicleType and fuelType)
            '汉': {'manufacturer': '比亚迪', 'model': '汉', 'vehicleType': '轿车', 'fuelType': '汽油'},
            '唐': {'manufacturer': '比亚迪', 'model': '唐', 'vehicleType': 'SUV', 'fuelType': '汽油'},
            '宋': {'manufacturer': '比亚迪', 'model': '宋', 'vehicleType': 'SUV', 'fuelType': '汽油'},
            '秦': {'manufacturer': '比亚迪', 'model': '秦', 'vehicleType': '轿车', 'fuelType': '汽油'},
            '元': {'manufacturer': '比亚迪', 'model': '元', 'vehicleType': 'SUV', 'fuelType': '汽油'},
            '汉EV': {'manufacturer': '比亚迪', 'model': '汉EV', 'vehicleType': '轿车', 'fuelType': '电动'},
            '唐DM': {'manufacturer': '比亚迪', 'model': '唐DM', 'vehicleType': 'SUV', 'fuelType': '混动'},

            # Tesla models (all electric sedans/SUVs)
            'Model 3': {'manufacturer': '特斯拉', 'model': 'Model 3', 'vehicleType': '轿车', 'fuelType': '电动'},
            'Model S': {'manufacturer': '特斯拉', 'model': 'Model S', 'vehicleType': '轿车', 'fuelType': '电动'},
            'Model X': {'manufacturer': '特斯拉', 'model': 'Model X', 'vehicleType': 'SUV', 'fuelType': '电动'},
            'Model Y': {'manufacturer': '特斯拉', 'model': 'Model Y', 'vehicleType': 'SUV', 'fuelType': '电动'},

            # BMW models (separate vehicleType and fuelType)
            '宝马3系': {'manufacturer': '宝马', 'model': '3系', 'vehicleType': '轿车', 'fuelType': '汽油'},
            '宝马5系': {'manufacturer': '宝马', 'model': '5系', 'vehicleType': '轿车', 'fuelType': '汽油'},
            '宝马X3': {'manufacturer': '宝马', 'model': 'X3', 'vehicleType': 'SUV', 'fuelType': '汽油'},
            '宝马X5': {'manufacturer': '宝马', 'model': 'X5', 'vehicleType': 'SUV', 'fuelType': '汽油'},
            '宝马i3': {'manufacturer': '宝马', 'model': 'i3', 'vehicleType': '轿车', 'fuelType': '电动'},

            # Mercedes models (separate vehicleType and fuelType)
            '奔驰C级': {'manufacturer': '奔驰', 'model': 'C级', 'vehicleType': '轿车', 'fuelType': '汽油'},
            '奔驰E级': {'manufacturer': '奔驰', 'model': 'E级', 'vehicleType': '轿车', 'fuelType': '汽油'},
            '奔驰GLC': {'manufacturer': '奔驰', 'model': 'GLC', 'vehicleType': 'SUV', 'fuelType': '汽油'},
            '奔驰GLE': {'manufacturer': '奔驰', 'model': 'GLE', 'vehicleType': 'SUV', 'fuelType': '汽油'},

            # Audi models (separate vehicleType and fuelType)
            '奥迪A4': {'manufacturer': '奥迪', 'model': 'A4', 'vehicleType': '轿车', 'fuelType': '汽油'},
            '奥迪A6': {'manufacturer': '奥迪', 'model': 'A6', 'vehicleType': '轿车', 'fuelType': '汽油'},
            '奥迪Q5': {'manufacturer': '奥迪', 'model': 'Q5', 'vehicleType': 'SUV', 'fuelType': '汽油'},
            '奥迪Q7': {'manufacturer': '奥迪', 'model': 'Q7', 'vehicleType': 'SUV', 'fuelType': '汽油'},

            # Toyota models (separate vehicleType and fuelType)
            '卡罗拉': {'manufacturer': '丰田', 'model': '卡罗拉', 'vehicleType': '轿车', 'fuelType': '汽油'},
            '凯美瑞': {'manufacturer': '丰田', 'model': '凯美瑞', 'vehicleType': '轿车', 'fuelType': '汽油'},
            '汉兰达': {'manufacturer': '丰田', 'model': '汉兰达', 'vehicleType': 'SUV', 'fuelType': '汽油'},
            'RAV4': {'manufacturer': '丰田', 'model': 'RAV4', 'vehicleType': 'SUV', 'fuelType': '汽油'},

            # Honda models (separate vehicleType and fuelType)
            '思域': {'manufacturer': '本田', 'model': '思域', 'vehicleType': '轿车', 'fuelType': '汽油'},
            '雅阁': {'manufacturer': '本田', 'model': '雅阁', 'vehicleType': '轿车', 'fuelType': '汽油'},
            'CR-V': {'manufacturer': '本田', 'model': 'CR-V', 'vehicleType': 'SUV', 'fuelType': '汽油'},
            '奥德赛': {'manufacturer': '本田', 'model': '奥德赛', 'vehicleType': 'MPV', 'fuelType': '汽油'},

            # Volkswagen models (separate vehicleType and fuelType)
            '速腾': {'manufacturer': '大众', 'model': '速腾', 'vehicleType': '轿车', 'fuelType': '汽油'},
            '迈腾': {'manufacturer': '大众', 'model': '迈腾', 'vehicleType': '轿车', 'fuelType': '汽油'},
            '途观': {'manufacturer': '大众', 'model': '途观', 'vehicleType': 'SUV', 'fuelType': '汽油'},

            # NIO models (all electric)
            'ES6': {'manufacturer': '蔚来', 'model': 'ES6', 'vehicleType': 'SUV', 'fuelType': '电动'},
            'ES8': {'manufacturer': '蔚来', 'model': 'ES8', 'vehicleType': 'SUV', 'fuelType': '电动'},
            'ET7': {'manufacturer': '蔚来', 'model': 'ET7', 'vehicleType': '轿车', 'fuelType': '电动'},

            # XPeng models (all electric)
            'P7': {'manufacturer': '小鹏', 'model': 'P7', 'vehicleType': '轿车', 'fuelType': '电动'},
            'G9': {'manufacturer': '小鹏', 'model': 'G9', 'vehicleType': 'SUV', 'fuelType': '电动'},

            # Li Auto models (all hybrid)
            '理想ONE': {'manufacturer': '理想', 'model': '理想ONE', 'vehicleType': 'SUV', 'fuelType': '混动'},
            '理想L9': {'manufacturer': '理想', 'model': '理想L9', 'vehicleType': 'SUV', 'fuelType': '混动'},

            # Add more models as metadata pool expands...
        }

        # Additional extractable fields (expandable) - English keys
        self.extractable_patterns = {
            'modelYear': [
                r'(\d{4})年款?',
                r'(\d{4})款',
                r'(20[0-9]{2})年?',
            ],
            'vehicleType': {
                'SUV': r'SUV|越野车|运动型多用途',
                '轿车': r'轿车|三厢车|sedan',
                '跑车': r'跑车|运动车|GT|敞篷',
                'MPV': r'MPV|商务车|七座|八座',
                '皮卡': r'皮卡|pickup|货车',
            },
            'fuelType': {
                '汽油': r'汽油|燃油|油车',
                '电动': r'电动车|纯电|新能源|EV|电池',
                '混动': r'混动|混合动力|PHEV|油电混合',
                '柴油': r'柴油|diesel',
            },
            'transmission': {
                '手动': r'手动|手挡|MT|手动变速',
                '自动': r'自动|自动挡|AT|自动变速',
                'CVT': r'CVT|无级变速',
                '双离合': r'双离合|DCT|DSG',
            },
            # NEW: Additional video metadata fields that can be extracted
            'author': {
                'pattern': r'author:([^|]+)',
                'extract_value': True
            },
            'views': {
                'pattern': r'views:([^|]+)',
                'extract_value': True
            },
            'source': {
                'pattern': r'source:([^|]+)',
                'extract_value': True
            },
            'description': {
                'pattern': r'desc:([^|]+)',
                'extract_value': True
            }
        }

    def extract_and_separate(self, raw_text: str) -> Tuple[Dict[str, Any], str]:
        """
        Extract recognized fields and return separated metadata + remaining text.

        Args:
            raw_text: Original concatenated metadata text

        Returns:
            (extracted_fields, remaining_text) where:
            - extracted_fields: Dict of recognized metadata
            - remaining_text: Text with recognized parts removed
        """

        extracted = {}
        remaining_text = raw_text
        extracted_spans = []  # Track what we've extracted to remove from original

        # 1. Extract vehicle models (most specific first)
        model_info = self._extract_vehicle_models(raw_text)
        if model_info:
            extracted.update(model_info)
            # Track extracted spans for removal
            for model_name in model_info.get('detectedModels', []):
                extracted_spans.extend(self._find_all_spans(raw_text, model_name))

        # 2. Extract manufacturers (if not already found via model)
        if not extracted.get('manufacturer'):
            manufacturer = self._extract_manufacturer(raw_text)
            if manufacturer:
                extracted['manufacturer'] = manufacturer
                # Find manufacturer mentions to remove
                for chinese_name, english_name in self.manufacturers.items():
                    if english_name == manufacturer:
                        extracted_spans.extend(self._find_all_spans(raw_text, chinese_name))

        # 3. Extract model year
        year = self._extract_model_year(raw_text)
        if year:
            extracted['modelYear'] = year
            # Find year mentions to remove
            year_patterns = [f'{year}年', f'{year}款', str(year)]
            for pattern in year_patterns:
                extracted_spans.extend(self._find_all_spans(raw_text, pattern))

        # 4. Extract category (if not from model)
        if not extracted.get('vehicleCategory'):
            category = self._extract_category(raw_text)
            if category:
                extracted['vehicleCategory'] = category
                # Find category mentions to remove
                for cat_name, pattern in self.extractable_patterns['vehicleCategory'].items():
                    if cat_name == category:
                        matches = re.finditer(pattern, raw_text, re.IGNORECASE)
                        extracted_spans.extend([(m.start(), m.end()) for m in matches])

        # 5. Extract additional fields
        fuel_type = self._extract_fuel_type(raw_text)
        if fuel_type:
            extracted['fuelType'] = fuel_type

        transmission = self._extract_transmission(raw_text)
        if transmission:
            extracted['transmission'] = transmission

        # 6. NEW: Extract video metadata fields (author, views, source, etc.)
        video_fields = self._extract_video_fields(raw_text)
        if video_fields:
            extracted.update(video_fields)
            # Add spans for these fields too
            for field_name, field_value in video_fields.items():
                if field_name == 'authorName':
                    extracted_spans.extend(self._find_all_spans(raw_text, f"author:{field_value}"))
                elif field_name == 'viewsText':
                    extracted_spans.extend(self._find_all_spans(raw_text, f"views:{field_value}"))
                elif field_name == 'sourcePlatform':
                    extracted_spans.extend(self._find_all_spans(raw_text, f"source:{field_value}"))
                elif field_name == 'descriptionText':
                    extracted_spans.extend(self._find_all_spans(raw_text, f"desc:{field_value}"))

        # 7. Remove extracted content from original text
        remaining_text = self._remove_extracted_spans(raw_text, extracted_spans)

        # 7. Clean up remaining text
        remaining_text = self._clean_remaining_text(remaining_text)

        return extracted, remaining_text

    def _extract_vehicle_models(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract vehicle models with highest priority."""

        detected_models = []
        best_match = None
        best_score = 0

        for model_name, model_info in self.vehicle_models.items():
            # Look for exact model matches
            if model_name.lower() in text.lower():
                detected_models.append(model_name)

                # Score based on model name length (longer = more specific)
                score = len(model_name)
                if score > best_score:
                    best_score = score
                    best_match = model_info.copy()
                    best_match['detectedModels'] = [model_name]

        if best_match:
            # Mark as structured extraction
            best_match['vehicleDetected'] = True
            best_match['structuredMatch'] = True
            best_match['extractionMethod'] = 'model_pattern'
            return best_match

        return None

    def _extract_manufacturer(self, text: str) -> Optional[str]:
        """Extract manufacturer if no model found."""

        text_lower = text.lower()
        for chinese_name, english_name in self.manufacturers.items():
            if chinese_name.lower() in text_lower:
                return english_name
        return None

    def _extract_model_year(self, text: str) -> Optional[int]:
        """Extract model year."""

        for pattern in self.extractable_patterns['modelYear']:
            match = re.search(pattern, text)
            if match:
                year = int(match.group(1))
                if 1990 <= year <= 2030:
                    return year
        return None

    def _extract_vehicle_type(self, text: str) -> Optional[str]:
        """Extract vehicle type (body style)."""

        for vehicle_type, pattern in self.extractable_patterns['vehicleType'].items():
            if re.search(pattern, text, re.IGNORECASE):
                return vehicle_type
        return None

    def _extract_fuel_type(self, text: str) -> Optional[str]:
        """Extract fuel type."""

        for fuel_type, pattern in self.extractable_patterns['fuelType'].items():
            if re.search(pattern, text, re.IGNORECASE):
                return fuel_type
        return None

    def _extract_transmission(self, text: str) -> Optional[str]:
        """Extract transmission type."""

        for trans_type, pattern in self.extractable_patterns['transmission'].items():
            if re.search(pattern, text, re.IGNORECASE):
                return trans_type
        return None

    def _extract_video_fields(self, text: str) -> Dict[str, str]:
        """Extract video metadata fields with English keys and RAW VALUES from original text."""

        video_fields = {}

        # Extract author (raw value)
        author_match = re.search(r'author:([^|]+)', text, re.IGNORECASE)
        if author_match:
            video_fields['authorName'] = author_match.group(1).strip()

        # Extract views (RAW NUMBER only)
        views_match = re.search(r'views:(\d+)', text, re.IGNORECASE)
        if views_match:
            video_fields['viewsCount'] = int(views_match.group(1))  # Store as integer

        # Extract source (ENUMERATED VALUE only: bilibili, youtube)
        source_match = re.search(r'source:(bilibili|youtube)', text, re.IGNORECASE)
        if source_match:
            video_fields['sourcePlatform'] = source_match.group(1).lower()  # bilibili or youtube

        return video_fields

    def _find_all_spans(self, text: str, pattern: str) -> List[Tuple[int, int]]:
        """Find all occurrences of pattern in text and return spans."""

        spans = []
        start = 0
        pattern_lower = pattern.lower()
        text_lower = text.lower()

        while True:
            pos = text_lower.find(pattern_lower, start)
            if pos == -1:
                break
            spans.append((pos, pos + len(pattern)))
            start = pos + 1

        return spans

    def _remove_extracted_spans(self, text: str, spans: List[Tuple[int, int]]) -> str:
        """Remove extracted spans from text."""

        if not spans:
            return text

        # Sort spans by start position (reverse order for removal)
        spans = sorted(set(spans), key=lambda x: x[0], reverse=True)

        result = text
        for start, end in spans:
            # Remove the span and clean up extra spaces
            before = result[:start].rstrip()
            after = result[end:].lstrip()

            # Handle separators
            if before and after:
                # Check if we need a separator
                if not before.endswith(('|', ',', '，', '：', ':')):
                    result = f"{before} | {after}"
                else:
                    result = f"{before} {after}"
            else:
                result = before + after

        return result

    def _clean_remaining_text(self, text: str) -> str:
        """Clean up remaining text after extraction."""

        # Remove redundant separators
        text = re.sub(r'\s*\|\s*\|\s*', ' | ', text)
        text = re.sub(r'\s*，\s*，\s*', '，', text)

        # Remove leading/trailing separators
        text = re.sub(r'^[\s|，：:]+', '', text)
        text = re.sub(r'[\s|，：:]+$', '', text)

        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()


class EnhancedTranscriptProcessor:
    """Enhanced processor with clean extraction-based metadata."""

    def __init__(self):
        self.metadata_extractor = MetadataExtractor()

    def process_transcript_chunks(
            self,
            transcript: str,
            video_metadata: Dict[str, Any],
            chunk_size: int = 1000,
            chunk_overlap: int = 200
    ) -> List[Document]:
        """Process transcript with extraction-based metadata (no duplication)."""

        # Create raw original field
        raw_original = self._create_raw_original_field(video_metadata)

        # Extract recognized fields and get remaining text
        extracted_fields, remaining_original = self.metadata_extractor.extract_and_separate(raw_original)

        # Split transcript
        chunks = self._split_transcript(transcript, chunk_size, chunk_overlap)

        # Process each chunk
        documents = []
        for i, chunk in enumerate(chunks):
            doc = self._create_enhanced_document(
                chunk=chunk,
                chunk_index=i,
                total_chunks=len(chunks),
                video_metadata=video_metadata,
                extracted_fields=extracted_fields,
                remaining_original=remaining_original
            )
            documents.append(doc)

        return documents

    def _create_raw_original_field(self, video_metadata: Dict[str, Any]) -> str:
        """Create raw original field with all video metadata using ENGLISH KEYS and RAW VALUES."""

        components = []

        # Title (no key needed, it's the main content) - WILL BE EXTRACTED
        title = video_metadata.get('title', '').strip()
        if title:
            components.append(title)

        # Author (English key, raw value)
        author = video_metadata.get('uploader', '').strip()
        if author:
            components.append(f"author:{author}")

        # Views (English key, RAW NUMBER - no Chinese formatting)
        views = video_metadata.get('view_count', 0)
        if views > 0:
            components.append(f"views:{views}")  # Raw number: views:553467

        # Description (no key, raw content) - WILL BE EXTRACTED  
        description = video_metadata.get('description', '').strip()
        if description:
            # Keep full description for extraction (will be truncated after extraction)
            components.append(description)

        # Source (English key, ENUMERATED VALUE)
        url = video_metadata.get('url', '')
        if 'bilibili.com' in url:
            components.append("source:bilibili")  # Use enumerated value
        elif 'youtube.com' in url:
            components.append("source:youtube")  # Use enumerated value

        return " | ".join(components)

    def _create_enhanced_document(
            self,
            chunk: str,
            chunk_index: int,
            total_chunks: int,
            video_metadata: Dict[str, Any],
            extracted_fields: Dict[str, Any],
            remaining_original: str
    ) -> Document:
        """Create document with extracted fields + remaining original (no duplication)."""

        # Build embedded metadata - EXTRACTED FIELDS ONLY (ENGLISH KEYS)
        metadata_parts = []

        # Add extracted vehicle fields with ENGLISH keys to save tokens
        if extracted_fields.get('manufacturer'):
            metadata_parts.append(f"【brand:{extracted_fields['manufacturer']}】")

        if extracted_fields.get('model'):
            metadata_parts.append(f"【model:{extracted_fields['model']}】")

        if extracted_fields.get('modelYear'):
            metadata_parts.append(f"【year:{extracted_fields['modelYear']}】")

        if extracted_fields.get('vehicleType'):
            metadata_parts.append(f"【type:{extracted_fields['vehicleType']}】")

        if extracted_fields.get('fuelType'):
            metadata_parts.append(f"【fuel:{extracted_fields['fuelType']}】")

        if extracted_fields.get('transmission'):
            metadata_parts.append(f"【trans:{extracted_fields['transmission']}】")

        # Add extracted video metadata fields (English keys, RAW VALUES)
        if extracted_fields.get('authorName'):
            metadata_parts.append(f"【author:{extracted_fields['authorName']}】")

        if extracted_fields.get('viewsCount'):
            metadata_parts.append(f"【views:{extracted_fields['viewsCount']}】")  # Raw number

        if extracted_fields.get('sourcePlatform'):
            metadata_parts.append(f"【source:{extracted_fields['sourcePlatform']}】")  # bilibili/youtube

        # Add REMAINING original (with extracted parts removed)
        if remaining_original.strip():
            # Truncate if still too long
            if len(remaining_original) > 100:
                truncated_remaining = remaining_original[:100] + "..."
            else:
                truncated_remaining = remaining_original
            metadata_parts.append(f"【other:{truncated_remaining}】")

        # Create embedded content
        if metadata_parts:
            embedded_text = f"{''.join(metadata_parts)}\n\n{chunk}"
        else:
            embedded_text = chunk

        # Comprehensive structured metadata
        structured_metadata = {
            # Basic video metadata (English keys)
            'source': 'bilibili' if 'bilibili.com' in video_metadata.get('url', '') else 'youtube',
            'sourceId': video_metadata.get('id', ''),
            'url': video_metadata.get('url', ''),
            'title': video_metadata.get('title', ''),
            'author': video_metadata.get('uploader', ''),
            'publishedDate': video_metadata.get('upload_date', ''),
            'duration': video_metadata.get('duration', 0),
            'viewCount': video_metadata.get('view_count', 0),
            'language': 'zh',

            # EXTRACTED vehicle metadata (English keys)
            'vehicleDetected': extracted_fields.get('vehicleDetected', False),
            'manufacturer': extracted_fields.get('manufacturer'),
            'vehicleModel': extracted_fields.get('model'),
            'modelYear': extracted_fields.get('modelYear'),
            'vehicleType': extracted_fields.get('vehicleType'),
            'fuelType': extracted_fields.get('fuelType'),
            'transmission': extracted_fields.get('transmission'),
            'structuredMatch': extracted_fields.get('structuredMatch', False),
            'extractionMethod': extracted_fields.get('extractionMethod', 'none'),

            # Extracted video metadata (English keys, RAW VALUES)
            'authorName': extracted_fields.get('authorName'),
            'viewsCount': extracted_fields.get('viewsCount'),  # Raw integer
            'sourcePlatform': extracted_fields.get('sourcePlatform'),  # bilibili/youtube

            # REMAINING original (with extracted parts removed) 
            'originalRemaining': remaining_original,
            'rawOriginal': self._create_raw_original_field(video_metadata),  # Keep full original for reference

            # Processing metadata
            'extractedFieldsCount': len([k for k, v in extracted_fields.items() if
                                         v and k not in ['vehicleDetected', 'structuredMatch', 'extractionMethod',
                                                         'detectedModels']]),
            'hasRemainingOriginal': bool(remaining_original.strip()),

            # Chunk metadata
            'chunkIndex': chunk_index,
            'totalChunks': total_chunks,
            'chunkId': f"{video_metadata.get('id', 'unknown')}_{chunk_index}",

            # UI flags
            'metadataInjected': bool(metadata_parts),
            'hasVehicleInfo': extracted_fields.get('vehicleDetected', False),
            'processingMethod': 'extraction_based_no_duplication',

            # Debug info
            'originalChunkLength': len(chunk),
            'enhancedChunkLength': len(embedded_text),
            'embeddedFieldsCount': len(metadata_parts)
        }

        return Document(
            page_content=embedded_text,
            metadata=structured_metadata
        )

    def _split_transcript(self, transcript: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split transcript with Chinese-aware separators."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', '。', '！', '？', '；', '，', ' ', '']
        )

        return text_splitter.split_text(transcript)


# Updated test and pattern matching for English keys
def test_extraction_system():
    """Test the extraction-based system with English keys (no duplication)."""

    processor = EnhancedTranscriptProcessor()

    # Test with rich metadata that should be partially extracted
    video_metadata = {
        'title': '2023款吉利星越L深度评测：动力性能与油耗测试',
        'id': 'test123',
        'url': 'https://www.bilibili.com/video/BV123456',
        'uploader': '汽车专业评测',
        'upload_date': '20231201',
        'duration': 1200,
        'view_count': 150000,  # Raw number
        'description': '全面测试2023款吉利星越L SUV的动力表现，包括2.0T发动机和自动变速箱的评测，以及真实油耗数据分析。'
    }

    # Test extraction on raw original
    raw_original = processor._create_raw_original_field(video_metadata)
    print("=== EXTRACTION TEST WITH RAW VALUES ===")
    print(f"Raw Original: {raw_original}")
    print("Note: Views are raw numbers (150000), source is enumerated (bilibili), title+description both extracted")

    extracted, remaining = processor.metadata_extractor.extract_and_separate(raw_original)
    print(f"\nExtracted Fields: {extracted}")
    print(f"Remaining Original (title + description remainders combined): {remaining}")

    # Show the improvements
    print(f"\nImprovements:")
    print(f"✅ Views: Raw number {extracted.get('viewsCount')} (not Chinese formatted)")
    print(f"✅ Source: Enumerated '{extracted.get('sourcePlatform')}' (not Chinese)")
    print(f"✅ Extraction: From both title AND description")
    print(f"✅ Remaining: Combined remainders from title + description + other fields")

    # Test with transcript
    test_transcript = "今天我们来测试这台星越L的实际表现。这台车搭载2.0T发动机配自动变速箱，官方说油耗是7.5L，我们来看看实际情况。经过一周的测试，这台SUV的表现还是不错的。"

    documents = processor.process_transcript_chunks(
        transcript=test_transcript,
        video_metadata=video_metadata,
        chunk_size=200,
        chunk_overlap=50
    )

    print(f"\n=== DOCUMENT RESULTS (ENGLISH KEYS) ===")
    print(f"Generated {len(documents)} documents")

    if documents:
        doc = documents[0]
        content = doc.page_content
        metadata = doc.metadata

        print(f"\nEmbedded Content (English Keys):")
        print(f"{content}")

        print(f"\nKey Metadata:")
        key_fields = ['vehicleDetected', 'manufacturer', 'vehicleModel', 'modelYear',
                      'vehicleCategory', 'fuelType', 'extractedFieldsCount',
                      'originalRemaining', 'hasRemainingOriginal']
        for field in key_fields:
            print(f"  {field}: {metadata.get(field)}")

        # Show what was extracted vs what remained
        print(f"\nExtraction Summary:")
        print(f"  Extracted fields: {metadata.get('extractedFieldsCount', 0)}")
        print(f"  Has remaining original: {metadata.get('hasRemainingOriginal', False)}")
        print(f"  Processing method: {metadata.get('processingMethod')}")

        # Analyze embedded patterns (now with English keys)
        embedded_patterns = re.findall(r'【[^】]+】', content)
        print(f"  Embedded patterns: {embedded_patterns}")

        # Token savings analysis
        print(f"\nToken Savings Analysis:")
        chinese_version = content.replace('brand:', '品牌:').replace('model:', '车型:').replace('year:',
                                                                                                '年份:').replace(
            'type:', '类型:').replace('fuel:', '燃料:').replace('trans:', '变速:').replace('other:', '其他:')
        token_savings = len(chinese_version) - len(content)
        savings_percent = (token_savings / len(chinese_version)) * 100 if chinese_version else 0
        print(f"  Estimated token savings: {token_savings} chars ({savings_percent:.1f}%)")


# Frontend analysis function updated for English embedded keys
def calculate_processing_statistics_english_embedded(documents: List[Dict]) -> Dict[str, Any]:
    """Calculate processing statistics for documents with English embedded keys."""

    if not documents:
        return {}

    total_docs = len(documents)
    stats = {
        'totalDocuments': total_docs,
        'documentsWithMetadata': 0,
        'documentsWithVehicleInfo': 0,
        'uniqueVehiclesDetected': [],
        'uniqueSources': set(),
        'metadataInjectionRate': 0.0,
        'vehicleDetectionRate': 0.0,
        'avgMetadataFieldsPerDoc': 0.0,
        'extractedFieldsUsage': 0.0,
        'tokenSavingsEstimate': 0
    }

    total_metadata_fields = 0
    total_extracted_fields = 0
    unique_vehicles = set()
    total_token_savings = 0

    # English key patterns for embedded metadata
    english_patterns = {
        'brand': r'【brand:[^】]+】',
        'model': r'【model:[^】]+】',
        'year': r'【year:[^】]+】',
        'type': r'【type:[^】]+】',
        'fuel': r'【fuel:[^】]+】',
        'trans': r'【trans:[^】]+】',
        'other': r'【other:[^】]+】'
    }

    for doc in documents:
        content = doc.get('content', doc.get('page_content', ''))
        metadata = doc.get('metadata', {})

        # Check for embedded metadata (English patterns)
        embedded_matches = re.findall(r'【[^】]+】', content)
        if embedded_matches:
            stats['documentsWithMetadata'] += 1
            total_metadata_fields += len(embedded_matches)

            # Calculate token savings vs Chinese keys
            chinese_equivalent = content
            chinese_equivalent = chinese_equivalent.replace('brand:', '品牌:')
            chinese_equivalent = chinese_equivalent.replace('model:', '车型:')
            chinese_equivalent = chinese_equivalent.replace('year:', '年份:')
            chinese_equivalent = chinese_equivalent.replace('type:', '类型:')
            chinese_equivalent = chinese_equivalent.replace('fuel:', '燃料:')
            chinese_equivalent = chinese_equivalent.replace('trans:', '变速:')
            chinese_equivalent = chinese_equivalent.replace('other:', '其他:')

            token_savings = len(chinese_equivalent) - len(content)
            total_token_savings += token_savings

        # Count extracted fields (English keys in structured metadata)
        extracted_count = metadata.get('extractedFieldsCount', 0)
        total_extracted_fields += extracted_count

        # Check vehicle info (English keys)
        if metadata.get('vehicleDetected'):
            stats['documentsWithVehicleInfo'] += 1

            # Track unique vehicles
            model = metadata.get('vehicleModel')
            manufacturer = metadata.get('manufacturer')
            if model and manufacturer:
                vehicle_name = f"{manufacturer} {model}"
                unique_vehicles.add(vehicle_name)
            elif model:
                unique_vehicles.add(model)

        # Track sources
        source = metadata.get('source')
        if source:
            stats['uniqueSources'].add(source)

    # Calculate rates
    if total_docs > 0:
        stats['metadataInjectionRate'] = stats['documentsWithMetadata'] / total_docs
        stats['vehicleDetectionRate'] = stats['documentsWithVehicleInfo'] / total_docs
        stats['extractedFieldsUsage'] = total_extracted_fields / total_docs

    if stats['documentsWithMetadata'] > 0:
        stats['avgMetadataFieldsPerDoc'] = total_metadata_fields / stats['documentsWithMetadata']

    stats['uniqueVehiclesDetected'] = list(unique_vehicles)
    stats['uniqueSources'] = list(stats['uniqueSources'])
    stats['tokenSavingsEstimate'] = total_token_savings

    return stats


if __name__ == "__main__":
    test_extraction_system()