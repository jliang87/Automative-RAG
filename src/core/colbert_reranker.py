from typing import Dict, List, Optional, Tuple, Union
import time
import os
import numpy as np

import torch
from langchain_core.documents import Document
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder

from src.config.settings import settings


class ColBERTReranker:
    """
    A GPU-optimized reranker that uses ColBERT v2 for late interaction retrieval
    with BGE-Reranker for enhanced bilingual performance.

    Uses a hybrid approach that combines:
    1. ColBERT v2 with token-level representations for fine-grained interaction
    2. BGE-Reranker for enhanced cross-lingual understanding
    3. Weighted combination of both scores for optimal bilingual performance
    """

    # Update the __init__ method in src/core/colbert_reranker.py to use local BGE reranker

    def __init__(
            self,
            model_name: str = "colbertv2.0",
            device: Optional[str] = None,
            max_query_length: int = 32,
            max_doc_length: int = 256,
            batch_size: int = 16,
            use_fp16: bool = True,
            similarity_metric: str = "maxsim",
            checkpoint_path: Optional[str] = None,
            use_bge_reranker: bool = True,
            colbert_weight: float = 0.8,
            bge_weight: float = 0.2,
            bge_model_name: str = "BAAI/bge-reranker-base"
    ):
        """
        Initialize the hybrid reranker with GPU optimizations.

        Args:
            model_name: Name of the ColBERT v2 model to use
            device: Device to run the model on (cuda or cpu)
            max_query_length: Maximum length of query tokens
            max_doc_length: Maximum length of document tokens
            batch_size: Batch size for document encoding
            use_fp16: Whether to use mixed precision (FP16) for faster computation
            similarity_metric: Similarity metric to use (maxsim or cosine)
            checkpoint_path: Optional custom path to model checkpoint
            use_bge_reranker: Whether to use BGE reranker for enhanced bilingual performance
            colbert_weight: Weight for ColBERT score in hybrid ranking (0.0-1.0)
            bge_weight: Weight for BGE score in hybrid ranking (0.0-1.0)
            bge_model_name: Name of the BGE reranker model to use
        """
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and self.device.startswith("cuda") and torch.cuda.is_available()
        self.similarity_metric = similarity_metric

        # Get model path (either from settings or provided checkpoint)
        model_path = checkpoint_path
        if not model_path and hasattr(settings, 'colbert_model_full_path'):
            model_path = settings.colbert_model_full_path
        self.model_path = model_path

        # Initialize ColBERT model and tokenizer
        self._initialize_model()

        # Setup automatic mixed precision for more efficient operations
        self.amp_enabled = self.use_fp16
        if self.amp_enabled:
            print(f"Using mixed precision (FP16) for reranker on {self.device}")

        # Hybrid reranking configuration
        self.use_bge_reranker = use_bge_reranker
        self.colbert_weight = colbert_weight
        self.bge_weight = bge_weight
        self.bge_model_name = bge_model_name

        # Initialize BGE reranker if enabled
        if self.use_bge_reranker:
            # Get BGE reranker path
            bge_model_path = None
            if hasattr(settings, 'bge_reranker_model_full_path'):
                bge_model_path = settings.bge_reranker_model_full_path

            print(f"Initializing BGE reranker: {bge_model_name}")
            print(f"BGE reranker local path: {bge_model_path}")

            try:
                print(f"Loading BGE reranker from local path: {bge_model_path}")
                self.bge_reranker = CrossEncoder(bge_model_path, device=self.device)
                print(f"BGE reranker loaded successfully on {self.device}")
            except Exception as e:
                print(f"Error loading BGE reranker: {str(e)}")
                self.use_bge_reranker = False

    def _initialize_model(self):
        """Initialize the model directly using transformers."""
        try:
            print(f"Loading model from {self.model_path or self.model_name}")

            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path or self.model_name,
                use_fast=True
            )

            # Initialize model
            self.model = AutoModel.from_pretrained(
                self.model_path or self.model_name
            ).to(self.device)

            # Convert to half precision if requested
            if self.use_fp16:
                self.model = self.model.half()

            # Set to evaluation mode
            self.model.eval()
            print(f"Successfully loaded model {self.model_name} with {self.model.config.hidden_size} hidden dimensions")

        except Exception as e:
            raise ValueError(f"Failed to load model. Error: {str(e)}")

    def _encode_query(self, query: str) -> torch.Tensor:
        """
        Encode the query for similarity comparison.

        Args:
            query: The query string

        Returns:
            Tensor of query embeddings
        """
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                # Tokenize the query
                tokens = self.tokenizer(
                    query,
                    add_special_tokens=True,
                    max_length=self.max_query_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                # Get the embeddings
                outputs = self.model(**tokens)

                # Use last hidden state (token embeddings)
                embeddings = outputs.last_hidden_state

                # Add batch dimension if not present
                if embeddings.dim() == 2:
                    embeddings = embeddings.unsqueeze(0)

                return embeddings

    def _encode_documents_batched(self, documents: List[Document]) -> List[torch.Tensor]:
        """
        Encode documents in batches.

        Args:
            documents: List of documents to encode

        Returns:
            List of document embeddings tensors
        """
        doc_embeddings_list = []

        # Extract document texts
        doc_texts = [doc.page_content for doc in documents]

        # Use smaller batches if we have very long documents
        effective_batch_size = max(1, self.batch_size // (self.max_doc_length // 128))

        # Process documents in batches
        for i in tqdm(range(0, len(doc_texts), effective_batch_size), desc="Encoding documents"):
            batch_texts = doc_texts[i:i + effective_batch_size]

            # Tokenize the batch
            batch_tokens = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=self.max_doc_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            # Get document embeddings
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    outputs = self.model(**batch_tokens)
                    batch_embeddings = outputs.last_hidden_state

            # Add each document's embeddings to the list
            for j in range(len(batch_texts)):
                doc_embeddings_list.append(batch_embeddings[j].detach())

        # Clear CUDA cache for large document collections
        if len(documents) > 1000:
            torch.cuda.empty_cache()

        return doc_embeddings_list

    def _compute_maxsim_scores(
            self, query_embeddings: torch.Tensor, doc_embeddings_list: List[torch.Tensor]
    ) -> List[float]:
        """
        Compute MaxSim scores between query and documents.

        Args:
            query_embeddings: Query token embeddings
            doc_embeddings_list: List of document token embeddings

        Returns:
            List of scores
        """
        scores = []

        # Use dynamic batch size based on available GPU memory
        effective_batch_size = min(self.batch_size, len(doc_embeddings_list))

        # Ensure query_embeddings has the right shape
        if query_embeddings.dim() == 3 and query_embeddings.size(0) == 1:
            # Remove batch dimension if batch size is 1
            query_emb = query_embeddings.squeeze(0)
        else:
            query_emb = query_embeddings

        # Process in batches
        for i in range(0, len(doc_embeddings_list), effective_batch_size):
            batch_embeddings = doc_embeddings_list[i:i + effective_batch_size]

            for doc_emb in batch_embeddings:
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    # Compute similarity matrix (q_len x d_len)
                    similarity = torch.matmul(query_emb, doc_emb.T)

                    # Get maximum similarity for each query token
                    max_sim, _ = similarity.max(dim=1)

                    # Filter out padding if we can detect it
                    # For BERT-like models, first token ([CLS]) and last non-padding token ([SEP])
                    # should be excluded from scoring
                    if max_sim.size(0) > 2:
                        # Exclude first and potentially last token, focus on content tokens
                        content_sim = max_sim[1:-1]
                        score = content_sim.sum().item()
                    else:
                        # If too short, use all tokens
                        score = max_sim.sum().item()

                scores.append(score)

        return scores

    def rerank(
            self, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Hybrid reranking using a weighted combination of ColBERT and BGE reranker.

        Args:
            query: The query string
            documents: List of documents to rerank
            top_k: Number of top documents to return

        Returns:
            List of (document, score) tuples sorted by score in descending order
        """
        if not documents:
            return []

        start_time = time.time()

        # Step 1: Get ColBERT scores
        colbert_results = self._colbert_rerank(query, documents)

        # If BGE reranker is not enabled, return ColBERT results
        if not self.use_bge_reranker:
            # Return top_k if specified
            if top_k is not None:
                colbert_results = colbert_results[:top_k]
            processing_time = time.time() - start_time
            print(f"ColBERT-only reranking completed in {processing_time:.2f}s")
            return colbert_results

        # Step 2: Extract documents and scores
        colbert_docs = [doc for doc, _ in colbert_results]
        colbert_scores = [score for _, score in colbert_results]

        # Step 3: Normalize ColBERT scores to [0, 1] range
        if colbert_scores:
            min_score = min(colbert_scores)
            max_score = max(colbert_scores)
            score_range = max_score - min_score
            if score_range > 0:
                colbert_scores = [(score - min_score) / score_range for score in colbert_scores]
            else:
                colbert_scores = [1.0 for _ in colbert_scores]

        # Step 4: Get BGE reranker scores
        # Prepare pairs for BGE reranker
        pairs = [[query, doc.page_content] for doc in colbert_docs]

        # Get scores from BGE reranker
        bge_scores = self.bge_reranker.predict(pairs)

        # Step 5: Normalize BGE scores to [0, 1] range
        if len(bge_scores) > 0:
            min_score = np.min(bge_scores)
            max_score = np.max(bge_scores)
            score_range = max_score - min_score
            if score_range > 0:
                bge_scores = [(score - min_score) / score_range for score in bge_scores]
            else:
                bge_scores = [1.0 for _ in bge_scores]

        # Step 6: Combine scores with weights
        combined_scores = [
            self.colbert_weight * colbert_scores[i] + self.bge_weight * bge_scores[i]
            for i in range(len(colbert_docs))
        ]

        # Step 7: Create final reranked results
        hybrid_results = list(zip(colbert_docs, combined_scores))

        # Step 8: Sort by combined score
        hybrid_results = sorted(hybrid_results, key=lambda x: x[1], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            hybrid_results = hybrid_results[:top_k]

        processing_time = time.time() - start_time
        print(f"Hybrid reranking completed in {processing_time:.2f}s")
        print(f"ColBERT weight: {self.colbert_weight:.2f}, BGE weight: {self.bge_weight:.2f}")

        return hybrid_results

    def _colbert_rerank(
            self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Original ColBERT reranking method, extracted as an internal method.

        Args:
            query: The query string
            documents: List of documents to rerank

        Returns:
            List of (document, score) tuples sorted by score in descending order
        """
        if not documents:
            return []

        start_time = time.time()

        # Encode query and documents
        query_embeddings = self._encode_query(query)
        doc_embeddings_list = self._encode_documents_batched(documents)

        # Score documents
        scores = self._compute_maxsim_scores(query_embeddings, doc_embeddings_list)

        # Create document-score pairs
        doc_score_pairs = list(zip(documents, scores))

        # Sort by score in descending order
        ranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        processing_time = time.time() - start_time
        print(f"ColBERT scoring completed in {processing_time:.2f}s for {len(documents)} documents")

        return ranked_docs

    def rerank_with_explanations(
            self, query: str, documents: List[Document], top_k: Optional[int] = None,
            num_explanations: int = 5
    ) -> List[Dict]:
        """
        Rerank documents and provide token-level explanations for matches.

        Args:
            query: The query string
            documents: List of documents to rerank
            top_k: Number of top documents to return
            num_explanations: Number of top token matches to explain

        Returns:
            List of dictionaries with document, score, and explanations
        """
        # First perform reranking (hybrid or ColBERT-only)
        reranked_docs = self.rerank(query, documents, top_k)

        # Extract just the documents (without scores)
        docs = [doc for doc, _ in reranked_docs]

        # Get explanations (from ColBERT only, as it provides token-level matches)
        # Note: We only explain using ColBERT since BGE-Reranker doesn't provide token-level explanations
        explanations = self._explain_colbert_matches(query, docs, num_explanations)

        return explanations

    def _explain_colbert_matches(
            self, query: str, documents: List[Document], num_explanations: int = 5
    ) -> List[Dict]:
        """
        Provide token-level explanations for ColBERT matches.

        Args:
            query: The query string
            documents: List of documents to explain
            num_explanations: Number of top token matches to explain

        Returns:
            List of dictionaries with document, score, and explanations
        """
        if not documents:
            return []

        # Get tokenized query for explanations
        query_encoding = self.tokenizer([query],
                                        add_special_tokens=True,
                                        max_length=self.max_query_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_tensors="pt")

        # Extract query tokens and masks
        query_token_ids = query_encoding.input_ids[0].tolist()
        query_token_strings = self.tokenizer.convert_ids_to_tokens(query_token_ids)
        query_mask = query_encoding.attention_mask[0].tolist()

        # Encode query
        query_embeddings = self._encode_query(query)

        # Ensure query_embeddings has the right shape
        if query_embeddings.dim() == 3 and query_embeddings.size(0) == 1:
            query_emb = query_embeddings.squeeze(0)
        else:
            query_emb = query_embeddings

        # Store document information and scores
        results = []

        # Process documents in batches
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]

            # Tokenize documents for explanation
            doc_encodings = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=self.max_doc_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            # Get document masks and tokens
            doc_masks = doc_encodings.attention_mask.tolist()
            doc_token_ids = doc_encodings.input_ids.tolist()

            # Encode documents
            batch_doc_embeddings = self._encode_documents_batched(
                [Document(page_content=text) for text in batch_texts]
            )

            # Process each document in the batch
            for j, doc in enumerate(batch_docs):
                # Get this document's embedding
                doc_embedding = batch_doc_embeddings[j]

                # Compute score and similarity matrix
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    # Calculate similarity matrix between query and document
                    similarity_matrix = torch.matmul(query_emb, doc_embedding.T)

                    # Get max similarities for each query token
                    max_similarities, max_indices = similarity_matrix.max(dim=1)

                    # Filter out padding and special tokens from scoring
                    content_indices = [
                        idx for idx, (mask, token) in enumerate(zip(query_mask, query_token_strings))
                        if mask == 1 and token not in ('[CLS]', '[SEP]')
                    ]

                    # Calculate total score using valid content tokens
                    total_score = sum(max_similarities[idx].item() for idx in content_indices)

                # Get document tokens for this document
                doc_token_strings = self.tokenizer.convert_ids_to_tokens(doc_token_ids[j])
                doc_mask = doc_masks[j]

                # Create explanations for meaningful query terms
                explanations = []
                for qidx in range(len(query_token_strings)):
                    # Skip padding, special tokens, and wordpiece continuations
                    if (query_mask[qidx] == 0 or
                            query_token_strings[qidx] in ('[PAD]', '[CLS]', '[SEP]', '[UNK]') or
                            query_token_strings[qidx].startswith('##')):
                        continue

                    # Get the index of the best matching document token
                    didx = max_indices[qidx].item()

                    # Make sure it's a valid token
                    if didx < len(doc_token_strings) and doc_mask[didx] == 1:
                        matched_doc_token = doc_token_strings[didx]
                        if matched_doc_token not in ('[PAD]', '[CLS]', '[SEP]', '[UNK]'):
                            # Get context around the matched token
                            context_start = max(0, didx - 2)
                            context_end = min(len(doc_token_strings), didx + 3)

                            # Filter out special tokens from context
                            context_tokens = [
                                t for t in doc_token_strings[context_start:context_end]
                                if t not in ('[PAD]', '[CLS]', '[SEP]', '[UNK]')
                            ]

                            # Join wordpiece tokens for better readability
                            context = ""
                            for t in context_tokens:
                                if t.startswith('##'):
                                    context = context[:-1] + t[2:]  # Remove space before wordpiece
                                else:
                                    context += t + " "

                            explanations.append({
                                "query_token": query_token_strings[qidx],
                                "doc_token": matched_doc_token,
                                "context": context.strip(),
                                "similarity": max_similarities[qidx].item()
                            })

                # Sort explanations by similarity score
                explanations.sort(key=lambda x: x["similarity"], reverse=True)

                # Add to results
                results.append({
                    "document": doc,
                    "score": total_score,
                    "explanations": explanations[:num_explanations]
                })

        # Sort results by score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results

    def batch_rerank_queries(
            self, queries: List[str], documents: List[Document], top_k: Optional[int] = None
    ) -> Dict[str, List[Tuple[Document, float]]]:
        """
        Efficiently rerank documents for multiple queries with shared document encoding.

        Args:
            queries: List of query strings
            documents: List of documents to rerank for each query
            top_k: Number of top documents to return per query

        Returns:
            Dictionary mapping each query to its ranked documents
        """
        if not documents or not queries:
            return {}

        start_time = time.time()

        # Encode documents once (this is the expensive part)
        doc_embeddings_list = self._encode_documents_batched(documents)

        # Process all queries
        results = {}

        for query in tqdm(queries, desc="Processing queries"):
            # Encode this query
            query_embedding = self._encode_query(query)

            # Score documents using shared embeddings
            scores = self._compute_maxsim_scores(query_embedding, doc_embeddings_list)

            # Create document-score pairs
            doc_score_pairs = list(zip(documents, scores))

            # Sort by score in descending order
            ranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

            # Apply BGE reranking if enabled
            if self.use_bge_reranker:
                # Get top documents for this query
                top_docs = ranked_docs[:top_k * 2] if top_k else ranked_docs  # Get 2x for reranking

                # Extract documents and scores
                colbert_docs = [doc for doc, _ in top_docs]
                colbert_scores = [score for _, score in top_docs]

                # Normalize ColBERT scores
                if colbert_scores:
                    min_score = min(colbert_scores)
                    max_score = max(colbert_scores)
                    score_range = max_score - min_score
                    if score_range > 0:
                        colbert_scores = [(score - min_score) / score_range for score in colbert_scores]
                    else:
                        colbert_scores = [1.0 for _ in colbert_scores]

                # Get BGE scores
                pairs = [[query, doc.page_content] for doc in colbert_docs]
                bge_scores = self.bge_reranker.predict(pairs)

                # Normalize BGE scores
                if len(bge_scores) > 0:
                    min_score = np.min(bge_scores)
                    max_score = np.max(bge_scores)
                    score_range = max_score - min_score
                    if score_range > 0:
                        bge_scores = [(score - min_score) / score_range for score in bge_scores]
                    else:
                        bge_scores = [1.0 for _ in bge_scores]

                # Combine scores
                combined_scores = [
                    self.colbert_weight * colbert_scores[i] + self.bge_weight * bge_scores[i]
                    for i in range(len(colbert_docs))
                ]

                # Create and sort hybrid results
                hybrid_results = list(zip(colbert_docs, combined_scores))
                hybrid_results = sorted(hybrid_results, key=lambda x: x[1], reverse=True)

                # Limit to top_k if specified
                if top_k:
                    hybrid_results = hybrid_results[:top_k]

                # Store results
                results[query] = hybrid_results
            else:
                # Limit to top_k if specified
                if top_k:
                    ranked_docs = ranked_docs[:top_k]

                # Store results for this query
                results[query] = ranked_docs

        processing_time = time.time() - start_time
        print(
            f"Batch reranking completed in {processing_time:.2f}s for {len(queries)} queries and {len(documents)} documents")

        return results