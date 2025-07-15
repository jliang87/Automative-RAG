"""
Guided Trust Loop - User Contribution Handler
Processes user contributions and manages the learning credit system
"""

import logging
import hashlib
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse

from ..models.validation_models import (
    ValidationStepType, ValidationStepResult, ValidationStatus,
    ContributionType, UserContribution, LearningCredit, ValidationUpdate
)
from .validation_cache_manager import ValidationCacheManager

logger = logging.getLogger(__name__)


class ContributionResult:
    """Result of processing a user contribution"""

    def __init__(self, contribution_accepted: bool, learning_credit: Optional[float] = None,
                 confidence_improvement: float = 0.0, new_validation_result: Optional[ValidationStepResult] = None,
                 contribution_impact: str = "", rejection_reason: str = ""):
        self.contribution_accepted = contribution_accepted
        self.learning_credit = learning_credit or 0.0
        self.confidence_improvement = confidence_improvement
        self.new_validation_result = new_validation_result
        self.contribution_impact = contribution_impact
        self.rejection_reason = rejection_reason


class ContributionHandler:
    """
    Handles user contributions and manages the guided trust loop
    """

    def __init__(self):
        self.cache_manager = ValidationCacheManager()

        # Initialize contribution processors
        self.contribution_processors = {
            ContributionType.URL_LINK: self._process_url_contribution,
            ContributionType.FILE_UPLOAD: self._process_file_contribution,
            ContributionType.DATABASE_LINK: self._process_database_contribution,
            ContributionType.TEXT_INPUT: self._process_text_contribution
        }

        # Initialize validation patterns for different contribution types
        self.validation_patterns = self._initialize_validation_patterns()

        # Learning credit calculation parameters
        self.credit_parameters = self._initialize_credit_parameters()

    def _initialize_validation_patterns(self) -> Dict[str, Any]:
        """Initialize patterns for validating different types of contributions"""
        return {
            "automotive_urls": {
                "official_manufacturers": [
                    r'toyota\.com', r'honda\.com', r'ford\.com', r'bmw\.com',
                    r'mercedes-benz\.com', r'audi\.com', r'volkswagen\.com',
                    r'nissan\.com', r'hyundai\.com', r'kia\.com', r'subaru\.com'
                ],
                "regulatory_sources": [
                    r'epa\.gov', r'nhtsa\.gov', r'dot\.gov', r'energy\.gov'
                ],
                "professional_automotive": [
                    r'edmunds\.com', r'kbb\.com', r'motortrend\.com',
                    r'caranddriver\.com', r'autoweek\.com', r'roadandtrack\.com',
                    r'automobilemag\.com', r'consumerreports\.org'
                ],
                "academic_sources": [
                    r'sae\.org', r'ieee\.org', r'\.edu', r'researchgate\.net'
                ]
            },
            "automotive_content": {
                "specifications": [
                    r'mpg', r'horsepower', r'torque', r'engine', r'transmission',
                    r'fuel economy', r'acceleration', r'top speed', r'weight'
                ],
                "features": [
                    r'features?', r'equipment', r'technology', r'infotainment',
                    r'safety', r'comfort', r'convenience'
                ],
                "ratings": [
                    r'nhtsa', r'iihs', r'safety rating', r'crash test',
                    r'reliability', r'jd power'
                ],
                "pricing": [
                    r'price', r'cost', r'msrp', r'invoice', r'value'
                ]
            },
            "file_types": {
                "acceptable": ['.pdf', '.doc', '.docx', '.txt', '.csv', '.xlsx'],
                "automotive_documents": [
                    r'specification', r'brochure', r'manual', r'review',
                    r'test', r'comparison', r'guide'
                ]
            }
        }

    def _initialize_credit_parameters(self) -> Dict[str, Any]:
        """Initialize parameters for learning credit calculation"""
        return {
            "base_credits": {
                ContributionType.URL_LINK: 5.0,
                ContributionType.FILE_UPLOAD: 8.0,
                ContributionType.DATABASE_LINK: 10.0,
                ContributionType.TEXT_INPUT: 3.0
            },
            "source_multipliers": {
                "official": 2.0,
                "regulatory": 2.5,
                "professional": 1.5,
                "academic": 2.0,
                "user_generated": 1.0,
                "other": 0.8
            },
            "validation_step_multipliers": {
                ValidationStepType.SOURCE_CREDIBILITY: 1.2,
                ValidationStepType.TECHNICAL_CONSISTENCY: 1.5,
                ValidationStepType.COMPLETENESS: 1.0,
                ValidationStepType.CONSENSUS: 1.3,
                ValidationStepType.RETRIEVAL: 0.8,
                ValidationStepType.LLM_INFERENCE: 1.1
            },
            "confidence_improvement_bonus": {
                "high": 5.0,  # >15% improvement
                "medium": 3.0,  # 5-15% improvement
                "low": 1.0  # <5% improvement
            }
        }

    async def process_contribution(self, job_id: str, step_type: str, contribution_data: Dict[str, Any],
                                   original_validation: Optional[ValidationStepResult] = None) -> ContributionResult:
        """
        Main entry point for processing user contributions
        """

        logger.info(f"Processing user contribution for job {job_id}, step {step_type}")

        try:
            # Parse contribution data
            contribution_type = ContributionType(contribution_data.get("type", "text_input"))
            contribution_content = contribution_data.get("content", {})
            user_id = contribution_data.get("user_id", "anonymous")

            # Create contribution record
            contribution = UserContribution(
                contribution_id=self._generate_contribution_id(job_id, user_id),
                user_id=user_id,
                validation_id=job_id,
                failed_step=ValidationStepType(step_type),
                contribution_type=contribution_type,
                source_url=contribution_content.get("url"),
                file_path=contribution_content.get("file_path"),
                text_content=contribution_content.get("text"),
                additional_context=contribution_content.get("context", ""),
                vehicle_scope=contribution_content.get("vehicle_scope", ""),
                submitted_at=datetime.now()
            )

            # Validate the contribution
            validation_result = await self._validate_contribution(contribution)

            if not validation_result["valid"]:
                return ContributionResult(
                    contribution_accepted=False,
                    rejection_reason=validation_result["reason"]
                )

            # Process the contribution based on type
            processor = self.contribution_processors[contribution_type]
            processing_result = await processor(contribution, step_type, contribution_content)

            if not processing_result["success"]:
                return ContributionResult(
                    contribution_accepted=False,
                    rejection_reason=processing_result["error"]
                )

            # Update knowledge base with contribution
            knowledge_update = await self._update_knowledge_base(
                contribution, processing_result["extracted_data"], step_type
            )

            # Retry validation with updated knowledge
            retry_result = await self._retry_validation_with_contribution(
                job_id, step_type, processing_result["extracted_data"], original_validation
            )

            if retry_result is None:
                return ContributionResult(
                    contribution_accepted=False,
                    rejection_reason="Could not retry validation with contribution"
                )

            # Calculate confidence improvement
            confidence_improvement = self._calculate_confidence_improvement(
                original_validation, retry_result
            )

            # Calculate learning credit
            learning_credit = self._calculate_learning_credit(
                contribution, step_type, confidence_improvement, processing_result
            )

            # Update contribution status
            contribution.status = "accepted"
            contribution.processed_at = datetime.now()

            # Create learning credit record
            credit_record = LearningCredit(
                credit_id=self._generate_credit_id(contribution.contribution_id),
                user_id=user_id,
                contribution_id=contribution.contribution_id,
                credit_points=learning_credit,
                credit_category=self._determine_credit_category(step_type, processing_result),
                impact_summary=f"Improved {step_type} validation by {confidence_improvement:.1f}%",
                benefits=self._generate_credit_benefits(step_type, confidence_improvement),
                validation_improvement=confidence_improvement,
                future_benefit_estimate=self._estimate_future_benefit(contribution, processing_result),
                earned_at=datetime.now()
            )

            logger.info(
                f"Contribution accepted: +{learning_credit:.1f} credits, +{confidence_improvement:.1f}% confidence")

            return ContributionResult(
                contribution_accepted=True,
                learning_credit=learning_credit,
                confidence_improvement=confidence_improvement,
                new_validation_result=retry_result,
                contribution_impact=f"Improved {step_type} validation with {contribution_type.value}"
            )

        except Exception as e:
            logger.error(f"Error processing contribution: {str(e)}")
            return ContributionResult(
                contribution_accepted=False,
                rejection_reason=f"Processing error: {str(e)}"
            )

    async def _validate_contribution(self, contribution: UserContribution) -> Dict[str, Any]:
        """Validate that a contribution is acceptable"""

        # Check contribution type and content
        if contribution.contribution_type == ContributionType.URL_LINK:
            return await self._validate_url_contribution(contribution)
        elif contribution.contribution_type == ContributionType.FILE_UPLOAD:
            return await self._validate_file_contribution(contribution)
        elif contribution.contribution_type == ContributionType.DATABASE_LINK:
            return await self._validate_database_contribution(contribution)
        elif contribution.contribution_type == ContributionType.TEXT_INPUT:
            return await self._validate_text_contribution(contribution)
        else:
            return {"valid": False, "reason": f"Unsupported contribution type: {contribution.contribution_type}"}

    async def _validate_url_contribution(self, contribution: UserContribution) -> Dict[str, Any]:
        """Validate URL contribution"""

        url = contribution.source_url
        if not url:
            return {"valid": False, "reason": "No URL provided"}

        # Parse URL
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {"valid": False, "reason": "Invalid URL format"}
        except Exception:
            return {"valid": False, "reason": "Cannot parse URL"}

        # Check if URL is from a known automotive source
        domain = parsed_url.netloc.lower()

        # Check against known automotive domains
        automotive_patterns = (
                self.validation_patterns["automotive_urls"]["official_manufacturers"] +
                self.validation_patterns["automotive_urls"]["regulatory_sources"] +
                self.validation_patterns["automotive_urls"]["professional_automotive"] +
                self.validation_patterns["automotive_urls"]["academic_sources"]
        )

        is_automotive_source = any(re.search(pattern, domain) for pattern in automotive_patterns)

        if not is_automotive_source:
            # Check if URL content might be automotive-related (lenient check)
            if not any(keyword in url.lower() for keyword in [
                'car', 'auto', 'vehicle', 'motor', 'drive', 'mpg', 'engine'
            ]):
                return {"valid": False, "reason": "URL does not appear to be automotive-related"}

        # Basic URL accessibility check (simplified)
        try:
            # In production, would make an actual HTTP request
            # For now, just validate URL format and domain
            if len(domain) < 3 or '.' not in domain:
                return {"valid": False, "reason": "URL domain appears invalid"}
        except Exception as e:
            return {"valid": False, "reason": f"Cannot verify URL accessibility: {str(e)}"}

        return {"valid": True, "reason": "URL appears valid and automotive-related"}

    async def _validate_file_contribution(self, contribution: UserContribution) -> Dict[str, Any]:
        """Validate file upload contribution"""

        file_path = contribution.file_path
        if not file_path:
            return {"valid": False, "reason": "No file provided"}

        # Check file extension
        file_ext = file_path.lower().split('.')[-1] if '.' in file_path else ''
        acceptable_extensions = [ext.lstrip('.') for ext in self.validation_patterns["file_types"]["acceptable"]]

        if file_ext not in acceptable_extensions:
            return {"valid": False, "reason": f"File type .{file_ext} not supported"}

        # Check filename for automotive relevance
        filename = file_path.lower()
        automotive_doc_patterns = self.validation_patterns["file_types"]["automotive_documents"]

        is_automotive_file = any(re.search(pattern, filename) for pattern in automotive_doc_patterns)

        if not is_automotive_file:
            # Check for automotive keywords in filename
            automotive_keywords = ['car', 'auto', 'vehicle', 'spec', 'mpg', 'review', 'test']
            if not any(keyword in filename for keyword in automotive_keywords):
                return {"valid": False, "reason": "File does not appear to be automotive-related"}

        # Simulate file accessibility check
        try:
            # In production, would check if file exists and is readable
            if len(file_path) < 5:
                return {"valid": False, "reason": "File path appears invalid"}
        except Exception as e:
            return {"valid": False, "reason": f"Cannot access file: {str(e)}"}

        return {"valid": True, "reason": "File appears valid and automotive-related"}

    async def _validate_database_contribution(self, contribution: UserContribution) -> Dict[str, Any]:
        """Validate database link contribution"""

        # Database links are typically pre-validated system integrations
        db_url = contribution.source_url
        if not db_url:
            return {"valid": False, "reason": "No database URL provided"}

        # Check for known automotive databases
        known_db_patterns = [
            r'epa\.gov', r'nhtsa\.gov', r'fueleconomy\.gov',
            r'cars\.com', r'edmunds\.com', r'kbb\.com'
        ]

        is_known_db = any(re.search(pattern, db_url) for pattern in known_db_patterns)

        if not is_known_db:
            return {"valid": False, "reason": "Database source not recognized as automotive authority"}

        return {"valid": True, "reason": "Database source appears to be automotive authority"}

    async def _validate_text_contribution(self, contribution: UserContribution) -> Dict[str, Any]:
        """Validate text input contribution"""

        text = contribution.text_content
        if not text or len(text.strip()) < 20:
            return {"valid": False, "reason": "Text contribution too short (minimum 20 characters)"}

        # Check for automotive content
        automotive_content = self.validation_patterns["automotive_content"]

        text_lower = text.lower()
        automotive_indicators = 0

        for category, patterns in automotive_content.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    automotive_indicators += 1
                    break  # Count each category once

        if automotive_indicators < 2:
            return {"valid": False, "reason": "Text does not contain sufficient automotive information"}

        # Check for specific data that could be useful
        has_specific_data = any([
            re.search(r'\d+\s*mpg', text_lower),
            re.search(r'\d+\s*hp', text_lower),
            re.search(r'\$\d+', text),
            re.search(r'\d+\s*star', text_lower),
            re.search(r'\d+/10', text)
        ])

        if not has_specific_data and len(text) < 100:
            return {"valid": False, "reason": "Text should contain specific automotive data or be more detailed"}

        return {"valid": True, "reason": "Text contains relevant automotive information"}

    async def _process_url_contribution(self, contribution: UserContribution, step_type: str,
                                        content: Dict[str, Any]) -> Dict[str, Any]:
        """Process URL contribution by fetching and extracting content"""

        url = contribution.source_url

        try:
            # Simulate URL content fetching
            # In production, would make actual HTTP request and extract content

            # Determine source type and authority
            source_type = self._classify_url_source_type(url)
            authority_score = self._calculate_url_authority_score(url, source_type)

            # Simulate extracted content based on URL type
            extracted_data = {
                "source_type": source_type,
                "authority_score": authority_score,
                "url": url,
                "content": f"Simulated content from {url}",
                "metadata": {
                    "source": url,
                    "authority_score": authority_score,
                    "fetched_at": datetime.now().isoformat(),
                    "extraction_method": "automated"
                }
            }

            # Add specific data based on step type
            if step_type == "technical_consistency":
                extracted_data["specifications"] = self._extract_simulated_specs(url)
            elif step_type == "source_credibility":
                extracted_data["credibility_indicators"] = self._extract_credibility_indicators(url)

            return {
                "success": True,
                "extracted_data": extracted_data,
                "processing_method": "url_fetch"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to process URL: {str(e)}"
            }

    async def _process_file_contribution(self, contribution: UserContribution, step_type: str,
                                         content: Dict[str, Any]) -> Dict[str, Any]:
        """Process file upload contribution"""

        file_path = contribution.file_path

        try:
            # Simulate file processing based on file type
            file_ext = file_path.lower().split('.')[-1] if '.' in file_path else ''

            if file_ext == 'pdf':
                extracted_data = await self._process_pdf_file(file_path, step_type)
            elif file_ext in ['doc', 'docx']:
                extracted_data = await self._process_document_file(file_path, step_type)
            elif file_ext in ['csv', 'xlsx']:
                extracted_data = await self._process_data_file(file_path, step_type)
            else:
                extracted_data = await self._process_text_file(file_path, step_type)

            extracted_data["source_type"] = "file_upload"
            extracted_data["authority_score"] = 0.7  # Default for user-uploaded files

            return {
                "success": True,
                "extracted_data": extracted_data,
                "processing_method": "file_processing"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to process file: {str(e)}"
            }

    async def _process_database_contribution(self, contribution: UserContribution, step_type: str,
                                             content: Dict[str, Any]) -> Dict[str, Any]:
        """Process database link contribution"""

        db_url = contribution.source_url

        try:
            # Simulate database query based on URL
            source_type = "regulatory" if "gov" in db_url else "professional"
            authority_score = 1.0 if "gov" in db_url else 0.9

            # Simulate database extraction
            extracted_data = {
                "source_type": source_type,
                "authority_score": authority_score,
                "database_url": db_url,
                "content": f"Official data from {db_url}",
                "metadata": {
                    "source": db_url,
                    "authority_score": authority_score,
                    "extracted_at": datetime.now().isoformat(),
                    "extraction_method": "database_query"
                }
            }

            # Add database-specific structured data
            if "epa.gov" in db_url:
                extracted_data["epa_data"] = self._simulate_epa_data(contribution.vehicle_scope)
            elif "nhtsa.gov" in db_url:
                extracted_data["safety_data"] = self._simulate_nhtsa_data(contribution.vehicle_scope)

            return {
                "success": True,
                "extracted_data": extracted_data,
                "processing_method": "database_query"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to process database link: {str(e)}"
            }

    async def _process_text_contribution(self, contribution: UserContribution, step_type: str,
                                         content: Dict[str, Any]) -> Dict[str, Any]:
        """Process text input contribution"""

        text = contribution.text_content

        try:
            # Extract structured information from text
            extracted_data = {
                "source_type": "user_generated",
                "authority_score": 0.6,  # Default for user text
                "content": text,
                "metadata": {
                    "source": "user_input",
                    "authority_score": 0.6,
                    "submitted_at": datetime.now().isoformat(),
                    "extraction_method": "text_analysis"
                }
            }

            # Extract specific information based on step type
            if step_type == "technical_consistency":
                extracted_data["extracted_specs"] = self._extract_specs_from_text(text)
            elif step_type == "source_credibility":
                extracted_data["expertise_indicators"] = self._extract_expertise_indicators(text)
            elif step_type == "completeness":
                extracted_data["information_categories"] = self._categorize_text_information(text)

            # Boost authority score if text shows expertise
            if self._text_shows_expertise(text):
                extracted_data["authority_score"] = 0.8

            return {
                "success": True,
                "extracted_data": extracted_data,
                "processing_method": "text_analysis"
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to process text: {str(e)}"
            }

    def _classify_url_source_type(self, url: str) -> str:
        """Classify the source type of a URL"""

        url_lower = url.lower()

        if any(pattern in url_lower for pattern in ['epa.gov', 'nhtsa.gov', 'dot.gov']):
            return "regulatory"
        elif any(pattern in url_lower for pattern in ['toyota.com', 'honda.com', 'ford.com']):
            return "official"
        elif any(pattern in url_lower for pattern in ['edmunds.com', 'motortrend.com', 'caranddriver.com']):
            return "professional"
        elif any(pattern in url_lower for pattern in ['.edu', 'sae.org', 'ieee.org']):
            return "academic"
        else:
            return "other"

    def _calculate_url_authority_score(self, url: str, source_type: str) -> float:
        """Calculate authority score for a URL"""

        authority_scores = {
            "regulatory": 1.0,
            "official": 0.95,
            "academic": 0.9,
            "professional": 0.8,
            "other": 0.6
        }

        base_score = authority_scores.get(source_type, 0.5)

        # Adjust for specific high-authority domains
        url_lower = url.lower()
        if 'epa.gov' in url_lower or 'nhtsa.gov' in url_lower:
            base_score = 1.0
        elif 'consumerreports.org' in url_lower:
            base_score = 0.95

        return base_score

    def _extract_simulated_specs(self, url: str) -> Dict[str, Any]:
        """Extract simulated specifications from URL (placeholder)"""

        return {
            "mpg_data": {"city": 25, "highway": 32, "combined": 28},
            "performance": {"horsepower": 200, "torque": 180},
            "extracted_from": url
        }

    def _extract_credibility_indicators(self, url: str) -> Dict[str, Any]:
        """Extract credibility indicators from URL"""

        return {
            "domain_authority": self._calculate_url_authority_score(url, self._classify_url_source_type(url)),
            "source_classification": self._classify_url_source_type(url),
            "bias_indicators": [],
            "expertise_level": "professional" if "professional" in self._classify_url_source_type(url) else "standard"
        }

    async def _process_pdf_file(self, file_path: str, step_type: str) -> Dict[str, Any]:
        """Process PDF file (simulated)"""

        return {
            "content": f"Extracted content from PDF: {file_path}",
            "document_type": "pdf",
            "pages_processed": 5,
            "specifications_found": ["mpg", "horsepower", "safety_rating"],
            "extracted_at": datetime.now().isoformat()
        }

    async def _process_document_file(self, file_path: str, step_type: str) -> Dict[str, Any]:
        """Process Word document file (simulated)"""

        return {
            "content": f"Extracted content from document: {file_path}",
            "document_type": "word_document",
            "sections_processed": 3,
            "automotive_content": True,
            "extracted_at": datetime.now().isoformat()
        }

    async def _process_data_file(self, file_path: str, step_type: str) -> Dict[str, Any]:
        """Process data file (CSV/Excel) (simulated)"""

        return {
            "content": f"Processed data file: {file_path}",
            "document_type": "data_file",
            "rows_processed": 100,
            "columns_found": ["vehicle", "mpg", "price", "rating"],
            "structured_data": True,
            "extracted_at": datetime.now().isoformat()
        }

    async def _process_text_file(self, file_path: str, step_type: str) -> Dict[str, Any]:
        """Process text file (simulated)"""

        return {
            "content": f"Processed text file: {file_path}",
            "document_type": "text_file",
            "lines_processed": 50,
            "automotive_keywords_found": 15,
            "extracted_at": datetime.now().isoformat()
        }

    def _simulate_epa_data(self, vehicle_scope: str) -> Dict[str, Any]:
        """Simulate EPA data extraction"""

        return {
            "vehicle": vehicle_scope,
            "city_mpg": 22,
            "highway_mpg": 30,
            "combined_mpg": 25,
            "fuel_type": "Regular Gasoline",
            "data_source": "EPA Fuel Economy Database"
        }

    def _simulate_nhtsa_data(self, vehicle_scope: str) -> Dict[str, Any]:
        """Simulate NHTSA data extraction"""

        return {
            "vehicle": vehicle_scope,
            "overall_rating": 5,
            "frontal_crash": 5,
            "side_crash": 4,
            "rollover": 4,
            "data_source": "NHTSA Safety Ratings"
        }

    def _extract_specs_from_text(self, text: str) -> Dict[str, Any]:
        """Extract specifications from text"""

        specs = {}

        # Extract MPG
        mpg_match = re.search(r'(\d+)\s*mpg', text, re.IGNORECASE)
        if mpg_match:
            specs["mpg"] = int(mpg_match.group(1))

        # Extract horsepower
        hp_match = re.search(r'(\d+)\s*(?:hp|horsepower)', text, re.IGNORECASE)
        if hp_match:
            specs["horsepower"] = int(hp_match.group(1))

        # Extract price
        price_match = re.search(r'\$(\d+,?\d+)', text)
        if price_match:
            specs["price"] = price_match.group(1)

        return specs

    def _extract_expertise_indicators(self, text: str) -> List[str]:
        """Extract indicators of expertise from text"""

        indicators = []

        expertise_patterns = [
            r'experience with', r'owned.*for.*years', r'mechanic', r'dealer',
            r'automotive engineer', r'test drove', r'compared to', r'in my opinion'
        ]

        for pattern in expertise_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                indicators.append(pattern)

        return indicators

    def _categorize_text_information(self, text: str) -> List[str]:
        """Categorize information types in text"""

        categories = []
        text_lower = text.lower()

        category_keywords = {
            "performance": ["horsepower", "acceleration", "speed", "handling"],
            "fuel_economy": ["mpg", "fuel", "gas", "efficiency"],
            "features": ["features", "equipment", "technology", "infotainment"],
            "reliability": ["reliability", "problems", "maintenance", "repair"],
            "safety": ["safety", "crash", "airbag", "rating"],
            "pricing": ["price", "cost", "value", "expensive", "affordable"]
        }

        for category, keywords in category_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                categories.append(category)

        return categories

    def _text_shows_expertise(self, text: str) -> bool:
        """Check if text shows automotive expertise"""

        expertise_indicators = [
            r'owned.*for.*years', r'mechanic', r'engineer', r'dealer',
            r'test drove', r'compared.*vehicles', r'automotive industry',
            r'professional.*opinion', r'technical.*specification'
        ]

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in expertise_indicators)

    async def _update_knowledge_base(self, contribution: UserContribution, extracted_data: Dict[str, Any],
                                     step_type: str) -> Dict[str, Any]:
        """Update knowledge base with contribution data"""

        try:
            # Simulate knowledge base update
            update_record = {
                "contribution_id": contribution.contribution_id,
                "step_type": step_type,
                "data_added": extracted_data,
                "vehicle_scope": contribution.vehicle_scope,
                "updated_at": datetime.now().isoformat(),
                "update_type": "user_contribution"
            }

            # In production, would actually update databases/knowledge stores
            logger.info(f"Updated knowledge base with contribution {contribution.contribution_id}")

            return {
                "success": True,
                "update_record": update_record
            }

        except Exception as e:
            logger.error(f"Failed to update knowledge base: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _retry_validation_with_contribution(self, job_id: str, step_type: str,
                                                  extracted_data: Dict[str, Any],
                                                  original_validation: Optional[ValidationStepResult]) -> Optional[
        ValidationStepResult]:
        """Retry validation step with contribution data"""

        try:
            # Get cached validation results to avoid re-running successful steps
            cached_results = await self.cache_manager.get_cached_validation_result(
                job_id, step_type, self._generate_documents_hash(extracted_data)
            )

            if cached_results:
                logger.info(f"Using cached validation result for {step_type}")
                return ValidationStepResult(**cached_results)

            # Simulate validation retry with enhanced data
            # In production, would re-run the specific validation step with updated data

            enhanced_result = self._simulate_enhanced_validation(
                step_type, extracted_data, original_validation
            )

            # Cache the new result
            await self.cache_manager.cache_validation_result(
                job_id, step_type, enhanced_result.__dict__,
                self._generate_documents_hash(extracted_data)
            )

            return enhanced_result

        except Exception as e:
            logger.error(f"Failed to retry validation: {str(e)}")
            return None

    def _simulate_enhanced_validation(self, step_type: str, extracted_data: Dict[str, Any],
                                      original_validation: Optional[ValidationStepResult]) -> ValidationStepResult:
        """Simulate enhanced validation with contribution data"""

        # Create improved validation result
        new_status = ValidationStatus.PASSED
        confidence_improvement = 15.0  # Base improvement

        # Adjust improvement based on contribution quality
        authority_score = extracted_data.get("authority_score", 0.5)
        confidence_improvement *= authority_score

        # Adjust based on step type
        step_multipliers = {
            "source_credibility": 1.2,
            "technical_consistency": 1.5,
            "completeness": 1.0,
            "consensus": 1.3
        }

        confidence_improvement *= step_multipliers.get(step_type, 1.0)

        enhanced_result = ValidationStepResult(
            step_id=f"{step_type}_enhanced_{datetime.now().isoformat()}",
            step_type=ValidationStepType(step_type),
            step_name=f"Enhanced {step_type.replace('_', ' ').title()}",
            status=new_status,
            confidence_impact=confidence_improvement,
            summary=f"Validation enhanced with user contribution (+{confidence_improvement:.1f}%)",
            details={
                "enhanced_with_contribution": True,
                "contribution_data": extracted_data,
                "original_validation": original_validation.__dict__ if original_validation else None,
                "enhancement_method": "user_contribution_integration"
            },
            started_at=datetime.now(),
            completed_at=datetime.now(),
            warnings=[],
            sources_used=[extracted_data.get("url", "user_contribution")]
        )

        return enhanced_result

    def _calculate_confidence_improvement(self, original_validation: Optional[ValidationStepResult],
                                          new_validation: ValidationStepResult) -> float:
        """Calculate confidence improvement from contribution"""

        if not original_validation:
            return new_validation.confidence_impact

        original_impact = original_validation.confidence_impact
        new_impact = new_validation.confidence_impact

        # Calculate improvement
        improvement = new_impact - original_impact

        # Ensure improvement is positive and reasonable
        return max(0.0, min(25.0, improvement))

    def _calculate_learning_credit(self, contribution: UserContribution, step_type: str,
                                   confidence_improvement: float, processing_result: Dict[str, Any]) -> float:
        """Calculate learning credit for user contribution"""

        # Base credit for contribution type
        base_credit = self.credit_parameters["base_credits"][contribution.contribution_type]

        # Source type multiplier
        source_type = processing_result["extracted_data"].get("source_type", "other")
        source_multiplier = self.credit_parameters["source_multipliers"].get(source_type, 1.0)

        # Validation step multiplier
        step_multiplier = self.credit_parameters["validation_step_multipliers"].get(
            ValidationStepType(step_type), 1.0
        )

        # Confidence improvement bonus
        if confidence_improvement > 15:
            improvement_bonus = self.credit_parameters["confidence_improvement_bonus"]["high"]
        elif confidence_improvement > 5:
            improvement_bonus = self.credit_parameters["confidence_improvement_bonus"]["medium"]
        else:
            improvement_bonus = self.credit_parameters["confidence_improvement_bonus"]["low"]

        # Calculate total credit
        total_credit = (base_credit * source_multiplier * step_multiplier) + improvement_bonus

        # Apply quality adjustments
        authority_score = processing_result["extracted_data"].get("authority_score", 0.5)
        total_credit *= authority_score

        return round(total_credit, 1)

    def _determine_credit_category(self, step_type: str, processing_result: Dict[str, Any]) -> str:
        """Determine the category of credit earned"""

        source_type = processing_result["extracted_data"].get("source_type", "other")

        category_mapping = {
            ("source_credibility", "regulatory"): "authority_verification",
            ("source_credibility", "official"): "manufacturer_validation",
            ("technical_consistency", "regulatory"): "official_specification",
            ("technical_consistency", "official"): "technical_validation",
            ("completeness", "professional"): "information_enhancement",
            ("consensus", "academic"): "research_contribution"
        }

        category = category_mapping.get((step_type, source_type))

        if not category:
            category = f"{step_type}_improvement"

        return category

    def _generate_credit_benefits(self, step_type: str, confidence_improvement: float) -> List[str]:
        """Generate list of benefits from the contribution"""

        benefits = [
            f"Improved {step_type.replace('_', ' ')} validation quality",
            f"Increased confidence by {confidence_improvement:.1f}%",
            "Enhanced knowledge base for future validations"
        ]

        if confidence_improvement > 15:
            benefits.append("Significant improvement to validation accuracy")

        if step_type == "technical_consistency":
            benefits.append("Added authoritative technical specifications")
        elif step_type == "source_credibility":
            benefits.append("Enhanced source authority assessment")

        return benefits

    def _estimate_future_benefit(self, contribution: UserContribution, processing_result: Dict[str, Any]) -> str:
        """Estimate future benefit of the contribution"""

        vehicle_scope = contribution.vehicle_scope
        source_type = processing_result["extracted_data"].get("source_type", "other")

        if source_type in ["regulatory", "official"]:
            return f"High - Will improve validation for {vehicle_scope} and similar vehicles"
        elif source_type == "professional":
            return f"Medium - Will enhance analysis quality for {vehicle_scope}"
        else:
            return f"Low-Medium - Will contribute to general knowledge about {vehicle_scope}"

    def _generate_contribution_id(self, job_id: str, user_id: str) -> str:
        """Generate unique contribution ID"""

        timestamp = datetime.now().isoformat()
        raw_id = f"{job_id}_{user_id}_{timestamp}"
        return hashlib.md5(raw_id.encode()).hexdigest()[:12]

    def _generate_credit_id(self, contribution_id: str) -> str:
        """Generate unique credit ID"""

        timestamp = datetime.now().isoformat()
        raw_id = f"credit_{contribution_id}_{timestamp}"
        return hashlib.md5(raw_id.encode()).hexdigest()[:12]

    def _generate_documents_hash(self, extracted_data: Dict[str, Any]) -> str:
        """Generate hash for document set including contribution"""

        content_key = json.dumps(extracted_data, sort_keys=True)
        return hashlib.md5(content_key.encode()).hexdigest()


# Global instance
contribution_handler = ContributionHandler()