import re
import jieba
import jieba.analyse
from typing import Dict, List, Optional, Tuple, Any
from langchain_core.documents import Document


class VehicleInfoExtractor:
    """Extract vehicle information from Chinese automotive content."""

    def __init__(self):
        # Initialize jieba with automotive keywords
        self._setup_automotive_dictionary()

        # Vehicle name patterns (Chinese automotive terms)
        self.vehicle_patterns = [
            # BMW models
            (r'宝马\s*([X1-9]|[1-8]系|i[3-8]|Z4)', 'BMW'),
            (r'BMW\s*([X1-9]|[1-8]|i[3-8]|Z4)', 'BMW'),

            # Mercedes models
            (r'奔驰\s*([A-Z]级|GLA|GLC|GLE|GLS|AMG)', 'Mercedes-Benz'),
            (r'Mercedes\s*([A-Z]\d*|GLA|GLC|GLE|GLS)', 'Mercedes-Benz'),

            # Audi models
            (r'奥迪\s*([A-Z]\d+|Q[2-8]|TT|R8)', 'Audi'),
            (r'Audi\s*([A-Z]\d+|Q[2-8]|TT|R8)', 'Audi'),

            # Tesla models
            (r'特斯拉\s*(Model\s*[SXYZ3]|Cybertruck)', 'Tesla'),
            (r'Tesla\s*(Model\s*[SXYZ3]|Cybertruck)', 'Tesla'),

            # Chinese brands - Geely
            (r'吉利\s*(星越L?|缤越|帝豪|博越)', 'Geely'),
            (r'Geely\s*(星越L?|缤越|帝豪|博越)', 'Geely'),

            # Chinese brands - BYD
            (r'比亚迪\s*(汉|唐|宋|秦|元)', 'BYD'),
            (r'BYD\s*(汉|唐|宋|秦|元)', 'BYD'),

            # Generic patterns for other brands
            (r'(丰田|Toyota)\s*(\w+)', 'Toyota'),
            (r'(本田|Honda)\s*(\w+)', 'Honda'),
            (r'(大众|Volkswagen|VW)\s*(\w+)', 'Volkswagen'),
            (r'(福特|Ford)\s*(\w+)', 'Ford'),
            (r'(日产|Nissan)\s*(\w+)', 'Nissan'),
            (r'(现代|Hyundai)\s*(\w+)', 'Hyundai'),
            (r'(起亚|Kia)\s*(\w+)', 'Kia'),
        ]

        # Regex for model year extraction
        self.year_patterns = [
            r'(\d{4})年款?',
            r'(\d{4})款',
            r'(20[0-9]{2})',
        ]

    def _setup_automotive_dictionary(self):
        """Setup jieba with automotive-specific keywords."""
        automotive_keywords = [
            # Vehicle types
            '轿车', 'SUV', '跑车', 'MPV', '皮卡', '卡车',

            # Brands (Chinese)
            '宝马', '奔驰', '奥迪', '丰田', '本田', '大众', '福特',
            '特斯拉', '吉利', '比亚迪', '长城', '奇瑞', '长安',

            # Models (Popular Chinese market models)
            '星越L', '缤越', '帝豪', '博越', '汉', '唐', '宋', '秦', '元',
            'Model3', 'ModelS', 'ModelX', 'ModelY',

            # Technical terms
            '发动机', '变速箱', '油耗', '马力', '扭矩', '加速', '制动',
            '续航', '充电', '电池', '混动', '纯电', '燃油',

            # Features
            '天窗', '座椅', '空调', '音响', '导航', '倒车影像',
            '自动驾驶', '辅助驾驶', '安全气囊', 'ESP', 'ABS',
        ]

        # Add keywords to jieba dictionary with high frequency
        for keyword in automotive_keywords:
            jieba.add_word(keyword, freq=1000)

    def extract_vehicle_info(self, title: str, description: str = "") -> Dict[str, Any]:
        """
        Extract vehicle information from title and description.

        Args:
            title: Video title
            description: Video description (optional)

        Returns:
            Dictionary with vehicle information
        """
        text = f"{title} {description}".strip()

        vehicle_info = {
            'name': None,
            'manufacturer': None,
            'year': None,
            'category': None,
            'confidence': 0.0
        }

        # Extract vehicle name and manufacturer
        best_match = self._find_best_vehicle_match(text)
        if best_match:
            vehicle_info.update(best_match)

        # Extract year
        year = self._extract_year(text)
        if year:
            vehicle_info['year'] = year

        # Extract category
        category = self._extract_category(text)
        if category:
            vehicle_info['category'] = category

        # Calculate confidence based on how much info we found
        confidence_factors = [
            vehicle_info['name'] is not None,
            vehicle_info['manufacturer'] is not None,
            vehicle_info['year'] is not None,
            vehicle_info['category'] is not None
        ]
        vehicle_info['confidence'] = sum(confidence_factors) / len(confidence_factors)

        return vehicle_info

    def _find_best_vehicle_match(self, text: str) -> Optional[Dict[str, str]]:
        """Find the best vehicle match in the text."""
        best_match = None
        best_score = 0

        for pattern, manufacturer in self.vehicle_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                # Score based on match length and position (earlier is better)
                score = len(match.group(0)) / (match.start() + 1)

                if score > best_score:
                    best_score = score
                    full_match = match.group(0).strip()

                    # Clean up the match
                    vehicle_name = self._clean_vehicle_name(full_match)

                    best_match = {
                        'name': vehicle_name,
                        'manufacturer': manufacturer,
                        'raw_match': full_match
                    }

        return best_match

    def _clean_vehicle_name(self, raw_name: str) -> str:
        """Clean and standardize vehicle name."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', '', raw_name)

        # Standardize some common variations
        standardizations = {
            'Model3': 'Model 3',
            'ModelS': 'Model S',
            'ModelX': 'Model X',
            'ModelY': 'Model Y',
            'BMW': '',  # Remove standalone brand mentions
            '宝马': '',
            '奔驰': '',
            '奥迪': '',
        }

        for old, new in standardizations.items():
            cleaned = cleaned.replace(old, new)

        return cleaned.strip()

    def _extract_year(self, text: str) -> Optional[int]:
        """Extract model year from text."""
        for pattern in self.year_patterns:
            match = re.search(pattern, text)
            if match:
                year = int(match.group(1))
                # Validate reasonable automotive year range
                if 1990 <= year <= 2030:
                    return year
        return None

    def _extract_category(self, text: str) -> Optional[str]:
        """Extract vehicle category from text."""
        categories = {
            'SUV': r'SUV|越野|运动型',
            '轿车': r'轿车|sedan|三厢',
            '跑车': r'跑车|sport|GT|敞篷',
            'MPV': r'MPV|商务|七座|八座',
            '皮卡': r'皮卡|pickup|货车',
            '新能源': r'电动|纯电|混动|新能源|EV|PHEV'
        }

        for category, pattern in categories.items():
            if re.search(pattern, text, re.IGNORECASE):
                return category

        return None


class EnhancedTranscriptProcessor:
    """Process transcripts with enhanced metadata injection."""

    def __init__(self):
        self.vehicle_extractor = VehicleInfoExtractor()

    def process_transcript_chunks(
        self,
        transcript: str,
        video_metadata: Dict[str, Any],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Process transcript into enhanced chunks with metadata injection.

        Args:
            transcript: Full transcript text
            video_metadata: Video metadata from yt-dlp
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks

        Returns:
            List of Document objects with enhanced metadata
        """
        # Extract vehicle information
        title = video_metadata.get('title', '')
        description = video_metadata.get('description', '')
        vehicle_info = self.vehicle_extractor.extract_vehicle_info(title, description)

        # Split transcript into chunks
        chunks = self._split_transcript(transcript, chunk_size, chunk_overlap)

        # Process each chunk
        documents = []
        for i, chunk in enumerate(chunks):
            # Create enhanced document
            doc = self._create_enhanced_document(
                chunk=chunk,
                chunk_index=i,
                total_chunks=len(chunks),
                video_metadata=video_metadata,
                vehicle_info=vehicle_info
            )
            documents.append(doc)

        return documents

    def _split_transcript(
        self,
        transcript: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Split transcript into overlapping chunks."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', '。', '！', '？', '；', '，', ' ', '']
        )

        return text_splitter.split_text(transcript)

    def _create_enhanced_document(
        self,
        chunk: str,
        chunk_index: int,
        total_chunks: int,
        video_metadata: Dict[str, Any],
        vehicle_info: Dict[str, Any]
    ) -> Document:
        """Create document with metadata injection."""

        # Build compact metadata prefix with English keys, Chinese values
        metadata_parts = []

        if vehicle_info.get('name'):
            metadata_parts.append(f"【model:{vehicle_info['name']}】")

        title = video_metadata.get('title', '')
        if title:
            # Truncate title if too long
            display_title = title[:30] + "..." if len(title) > 30 else title
            metadata_parts.append(f"【title:{display_title}】")

        if vehicle_info.get('year'):
            metadata_parts.append(f"【year:{vehicle_info['year']}】")

        if vehicle_info.get('category'):
            metadata_parts.append(f"【category:{vehicle_info['category']}】")

        # Inject metadata into chunk text
        metadata_prefix = ''.join(metadata_parts)
        embedded_text = f"{metadata_prefix}{chunk}" if metadata_parts else chunk

        # Build structured metadata (English keys, Chinese values)
        structured_metadata = {
            # Basic video metadata
            'source': 'bilibili' if 'bilibili.com' in video_metadata.get('url', '') else 'youtube',
            'source_id': video_metadata.get('id', ''),
            'url': video_metadata.get('url', ''),
            'title': title,
            'author': video_metadata.get('uploader', ''),
            'published_date': video_metadata.get('upload_date', ''),
            'duration': video_metadata.get('duration', 0),
            'view_count': video_metadata.get('view_count', 0),
            'language': 'zh',  # Assuming Chinese content

            # Vehicle information (Chinese values for models)
            'manufacturer': vehicle_info.get('manufacturer'),
            'model': vehicle_info.get('name'),  # Chinese model name like "星越L"
            'year': vehicle_info.get('year'),
            'category': vehicle_info.get('category'),
            'vehicle_confidence': vehicle_info.get('confidence', 0.0),

            # Chunk information
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
            'chunk_id': f"{video_metadata.get('id', 'unknown')}_{chunk_index}",

            # Processing metadata
            'document_type': 'chunk_with_metadata',
            'search_type': 'semantic',
            'has_vehicle_info': vehicle_info.get('name') is not None,
            'metadata_injected': bool(metadata_parts),

            # For debugging
            'original_chunk_length': len(chunk),
            'enhanced_chunk_length': len(embedded_text),
            'metadata_prefix': metadata_prefix
        }

        return Document(
            page_content=embedded_text,
            metadata=structured_metadata
        )


class ResultScoreNormalizer:
    """Normalize reranked document scores for accurate UI display."""

    @staticmethod
    def normalize_reranked_results(
        reranked_docs: List[Tuple[Document, float]],
        original_scores: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Normalize reranked scores and format for UI display.

        Args:
            reranked_docs: List of (document, hybrid_score) tuples from reranking
            original_scores: Optional mapping of doc_id -> original_vector_score

        Returns:
            List of formatted document dictionaries with normalized scores
        """
        if not reranked_docs:
            return []

        # Extract scores for normalization
        scores = [score for _, score in reranked_docs]

        # Normalize scores to 0-1 range
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        formatted_docs = []

        for i, (doc, hybrid_score) in enumerate(reranked_docs):
            # Normalize score to 0-1 range
            if score_range > 0:
                normalized_score = (hybrid_score - min_score) / score_range
            else:
                normalized_score = 1.0  # All scores are the same

            # Get original vector score if available
            doc_id = doc.metadata.get('chunk_id', f'doc_{i}')
            original_score = None
            if original_scores and doc_id in original_scores:
                original_score = original_scores[doc_id]

            # Format for UI
            formatted_doc = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': normalized_score,  # ✅ Final score shown to user
                'raw_hybrid_score': hybrid_score,     # For debugging
                'original_vector_score': original_score,  # Optional, for comparison
                'rerank_position': i + 1,
                'score_normalized': True
            }

            formatted_docs.append(formatted_doc)

        return formatted_docs

    @staticmethod
    def apply_score_boost_for_vehicle_match(
        formatted_docs: List[Dict[str, Any]],
        query_vehicle_info: Optional[Dict[str, Any]] = None,
        boost_factor: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Apply score boost for documents matching query vehicle information.

        Args:
            formatted_docs: Formatted documents from normalize_reranked_results
            query_vehicle_info: Vehicle info extracted from query
            boost_factor: How much to boost matching scores (0.1 = 10% boost)

        Returns:
            Documents with potentially boosted scores
        """
        if not query_vehicle_info or not formatted_docs:
            return formatted_docs

        query_model = query_vehicle_info.get('name')
        query_manufacturer = query_vehicle_info.get('manufacturer')

        for doc in formatted_docs:
            metadata = doc['metadata']
            doc_model = metadata.get('model')
            doc_manufacturer = metadata.get('manufacturer')

            # Apply boost if vehicle matches
            if query_model and doc_model and query_model == doc_model:
                # Exact model match - strong boost
                original_score = doc['relevance_score']
                boosted_score = min(1.0, original_score + boost_factor * 2)
                doc['relevance_score'] = boosted_score
                doc['score_boosted'] = True
                doc['boost_reason'] = f'exact_model_match:{query_model}'

            elif query_manufacturer and doc_manufacturer and query_manufacturer == doc_manufacturer:
                # Manufacturer match - smaller boost
                original_score = doc['relevance_score']
                boosted_score = min(1.0, original_score + boost_factor)
                doc['relevance_score'] = boosted_score
                doc['score_boosted'] = True
                doc['boost_reason'] = f'manufacturer_match:{query_manufacturer}'

        # Re-sort by boosted scores
        formatted_docs.sort(key=lambda x: x['relevance_score'], reverse=True)

        # Update rerank positions
        for i, doc in enumerate(formatted_docs):
            doc['rerank_position'] = i + 1

        return formatted_docs


# Usage example integration with existing video processing
def integrate_with_video_processing():
    """Example of how to integrate with existing video processing pipeline."""

    # This would be called from transcribe_video_task in job_chain.py
    processor = EnhancedTranscriptProcessor()
    normalizer = ResultScoreNormalizer()

    # Example usage in video processing
    def enhanced_video_processing(transcript: str, video_metadata: Dict[str, Any]) -> List[Document]:
        """Enhanced video processing with metadata injection."""

        # Process transcript with metadata injection
        documents = processor.process_transcript_chunks(
            transcript=transcript,
            video_metadata=video_metadata,
            chunk_size=1000,
            chunk_overlap=200
        )

        return documents

    # Example usage in retrieval/reranking
    def enhanced_document_retrieval(query: str, reranked_docs: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
        """Enhanced document retrieval with score normalization."""

        # Extract vehicle info from query
        extractor = VehicleInfoExtractor()
        query_vehicle_info = extractor.extract_vehicle_info(query)

        # Normalize scores
        formatted_docs = normalizer.normalize_reranked_results(reranked_docs)

        # Apply vehicle matching boost
        if query_vehicle_info.get('confidence', 0) > 0.5:
            formatted_docs = normalizer.apply_score_boost_for_vehicle_match(
                formatted_docs,
                query_vehicle_info,
                boost_factor=0.15
            )

        return formatted_docs

    return enhanced_video_processing, enhanced_document_retrieval


if __name__ == "__main__":
    # Test the enhanced transcript processor
    processor = EnhancedTranscriptProcessor()

    # Test data
    test_transcript = """
    今天我们来测试一下2023款吉利星越L的性能表现。
    这款SUV在动力方面搭载了2.0T涡轮增压发动机，
    最大功率218马力，峰值扭矩325牛·米。
    在实际驾驶中，星越L的加速表现非常出色，
    0-100公里加速仅需7.9秒。
    """

    test_metadata = {
        'title': '2023款吉利星越L深度评测：性能、配置、油耗全面解析',
        'id': 'test_video_123',
        'url': 'https://www.bilibili.com/video/BV1234567890',
        'uploader': '汽车评测频道',
        'upload_date': '20231201',
        'duration': 1800,
        'view_count': 50000
    }

    # Process transcript
    documents = processor.process_transcript_chunks(test_transcript, test_metadata)

    # Print results
    print(f"Generated {len(documents)} enhanced documents:")
    for i, doc in enumerate(documents):
        print(f"\n--- Document {i+1} ---")
        print(f"Content: {doc.page_content[:100]}...")
        print(f"Vehicle: {doc.metadata.get('model')} ({doc.metadata.get('manufacturer')})")
        print(f"Metadata injected: {doc.metadata.get('metadata_injected')}")
        print(f"Vehicle confidence: {doc.metadata.get('vehicle_confidence'):.2f}")

    # Expected output with English keys:
    # Content: 【model:星越L】【title:2023款吉利星越L深度评测：性能、配置、油耗全面解析】今天我们来测试一下2023款吉利星越L的性能表现...