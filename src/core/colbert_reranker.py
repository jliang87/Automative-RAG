from typing import Dict, List, Optional, Tuple, Union
import time

import torch
from colbert_ai import ColBERT
from langchain_core.documents import Document
from tqdm import tqdm
from transformers import AutoTokenizer


class ColBERTReranker:
    """
    A GPU-optimized reranker that uses ColBERT for late interaction retrieval.
    
    ColBERT preserves token-level representations and performs fine-grained
    interaction between query and document tokens, enabling more precise
    semantic matching with efficient GPU utilization.
    """

    def __init__(
        self,
        model_name: str = "colbert-ir/colbertv2.0",
        device: Optional[str] = None,
        max_query_length: int = 32,
        max_doc_length: int = 256,
        batch_size: int = 16,
        use_fp16: bool = True,
    ):
        """
        Initialize the ColBERT reranker with GPU optimizations.

        Args:
            model_name: Name of the ColBERT model to use
            device: Device to run the model on (cuda or cpu)
            max_query_length: Maximum length of query tokens
            max_doc_length: Maximum length of document tokens
            batch_size: Batch size for document encoding
            use_fp16: Whether to use mixed precision (FP16) for faster computation
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and self.device == "cuda" and torch.cuda.is_available()
        
        # Initialize the ColBERT model
        self.colbert = ColBERT.load(
            model_name,
            index_root=None,  # We don't need index functionality
            device=self.device,
        )
        
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Setup automatic mixed precision if requested
        self.amp_enabled = self.use_fp16
        if self.amp_enabled:
            print(f"Using mixed precision (FP16) for ColBERT reranker on {self.device}")

    def _encode_query(self, query: str) -> torch.Tensor:
        """
        Encode the query using ColBERT with GPU acceleration.
        
        Args:
            query: The query string
            
        Returns:
            Tensor of query embeddings
        """
        # Tokenize the query
        query_tokens = self.tokenizer(
            query,
            add_special_tokens=True,
            max_length=self.max_query_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Get query embeddings with mixed precision if enabled
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                query_embeddings = self.colbert.query(**query_tokens)
            
        return query_embeddings

    def _encode_documents_batched(self, documents: List[Document]) -> List[torch.Tensor]:
        """
        Encode documents in batches using GPU acceleration.
        
        Args:
            documents: List of documents to encode
            
        Returns:
            List of document embeddings tensors
        """
        doc_embeddings_list = []
        
        # Process documents in batches
        for i in tqdm(range(0, len(documents), self.batch_size), desc="Encoding documents with ColBERT"):
            batch_docs = documents[i:i + self.batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]
            
            # Tokenize the batch
            batch_tokens = self.tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=self.max_doc_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            
            # Get document embeddings with mixed precision if enabled
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    batch_embeddings = self.colbert.doc(
                        input_ids=batch_tokens.input_ids,
                        attention_mask=batch_tokens.attention_mask
                    )
                
            # Add each document's embeddings to the list
            doc_embeddings_list.extend([emb for emb in batch_embeddings])
                
        return doc_embeddings_list

    def _score_documents_batched(
        self, query_embeddings: torch.Tensor, doc_embeddings_list: List[torch.Tensor]
    ) -> List[float]:
        """
        Score documents in batches using the MaxSim operator with GPU acceleration.
        
        Args:
            query_embeddings: Query token embeddings
            doc_embeddings_list: List of document token embeddings
            
        Returns:
            List of scores
        """
        scores = []
        
        # Process in batches to avoid GPU memory issues
        for i in range(0, len(doc_embeddings_list), self.batch_size):
            batch_embeddings = doc_embeddings_list[i:i + self.batch_size]
            
            # Stack document embeddings for batch processing
            # Each document has shape [1, doc_length, dim]
            stacked_docs = torch.cat(batch_embeddings, dim=0)
            
            # Expand query embeddings to match batch size
            # From [1, query_length, dim] to [batch_size, query_length, dim]
            batch_size = stacked_docs.shape[0]
            expanded_query = query_embeddings.expand(batch_size, -1, -1)
            
            # Compute scores in a single batch operation using mixed precision
            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                # Compute similarity matrices for all documents in batch
                # Shape: [batch_size, query_length, doc_length]
                similarity_matrices = torch.bmm(
                    expanded_query, 
                    stacked_docs.transpose(1, 2)
                )
                
                # For each query token, get maximum similarity across document tokens
                # Shape: [batch_size, query_length]
                max_similarities, _ = similarity_matrices.max(dim=2)
                
                # Sum the maximum similarities for each document
                # Shape: [batch_size]
                batch_scores = max_similarities.sum(dim=1)
            
            # Convert to list and add to overall scores
            scores.extend(batch_scores.cpu().tolist())
            
        return scores

    def rerank(
        self, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using ColBERT's late interaction scoring with GPU acceleration.
        
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
            
        # Encode query and documents using batched processing
        query_embeddings = self._encode_query(query)
        doc_embeddings_list = self._encode_documents_batched(documents)
        
        # Score documents using batched processing
        scores = self._score_documents_batched(query_embeddings, doc_embeddings_list)
        
        # Create document-score pairs
        doc_score_pairs = list(zip(documents, scores))
        
        # Sort by score in descending order
        ranked_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            ranked_docs = ranked_docs[:top_k]
        
        processing_time = time.time() - start_time
        print(f"ColBERT reranking completed in {processing_time:.2f}s for {len(documents)} documents")
        
        return ranked_docs

    def rerank_with_explanations(
        self, query: str, documents: List[Document], top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Rerank documents and provide token-level explanations for matches.
        
        Args:
            query: The query string
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of dictionaries with document, score, and explanations
        """
        if not documents:
            return []
            
        # Encode query
        query_embeddings = self._encode_query(query)
        
        # Tokenize query for explanations
        query_tokens = self.tokenizer(
            query,
            add_special_tokens=True,
            max_length=self.max_query_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        query_token_ids = query_tokens.input_ids[0].tolist()
        query_token_strings = self.tokenizer.convert_ids_to_tokens(query_token_ids)
        
        results = []
        
        # Process documents in batches
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            batch_texts = [doc.page_content for doc in batch_docs]
            
            # Tokenize the documents
            doc_tokens = self.tokenizer(
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
                    doc_embeddings = self.colbert.doc(
                        input_ids=doc_tokens.input_ids,
                        attention_mask=doc_tokens.attention_mask
                    )
            
            # Process each document in the batch
            for j, doc in enumerate(batch_docs):
                # Extract token strings for this document
                doc_token_ids = doc_tokens.input_ids[j].tolist()
                doc_token_strings = self.tokenizer.convert_ids_to_tokens(doc_token_ids)
                
                # Get embeddings for this specific document
                doc_embedding = doc_embeddings[j].unsqueeze(0)  # Add batch dimension back
                
                # Calculate similarity matrix
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    similarity_matrix = torch.matmul(
                        query_embeddings, 
                        doc_embedding.transpose(1, 2)
                    ).squeeze(0)
                
                # Get max similarities and their positions
                max_similarities, max_indices = similarity_matrix.max(dim=1)
                total_score = max_similarities.sum().item()
                
                # Create explanations for top query terms
                explanations = []
                for qidx in range(min(len(query_token_strings), self.max_query_length)):
                    if query_token_strings[qidx].startswith('##') or query_token_strings[qidx] in ('[PAD]', '[CLS]', '[SEP]'):
                        continue
                        
                    didx = max_indices[qidx].item()
                    if didx < len(doc_token_strings):
                        matched_doc_token = doc_token_strings[didx]
                        if matched_doc_token not in ('[PAD]', '[CLS]', '[SEP]'):
                            explanations.append({
                                "query_token": query_token_strings[qidx],
                                "doc_token": matched_doc_token,
                                "similarity": max_similarities[qidx].item()
                            })
                
                # Sort explanations by similarity score
                explanations.sort(key=lambda x: x["similarity"], reverse=True)
                
                # Add to results
                results.append({
                    "document": doc,
                    "score": total_score,
                    "explanations": explanations[:5]  # Top 5 explanations
                })
        
        # Sort results by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to top_k if specified
        if top_k is not None:
            results = results[:top_k]
            
        return results