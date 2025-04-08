import numpy as np
from typing import List, Dict

class HybridRanker:
    def __init__(self, bm25_weight: float = 0.5, vector_weight: float = 0.5):
        """
        Initialize ranker with weights for combining BM25 and vector scores.
        
        Args:
            bm25_weight: Weight for BM25 scores (0.0 to 1.0)
            vector_weight: Weight for vector search scores (0.0 to 1.0)
        """
        if not np.isclose(bm25_weight + vector_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")
            
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    @staticmethod
    def _normalize(scores: List[float]) -> List[float]:
        """Min-max normalize scores to [0, 1] range"""
        if not scores:
            return []
            
        min_val, max_val = min(scores), max(scores)
        if min_val == max_val:
            return [0.5] * len(scores)  # Handle uniform scores
            
        return [(x - min_val) / (max_val - min_val) for x in scores]

    def rank_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Combine and rank documents based on BM25 and vector scores.
        
        Args:
            documents: List of dicts with 'bm25_score' and 'vector_score' keys
            
        Returns:
            List of documents sorted by combined score (descending)
        """
        if not documents:
            return []
            
        # Extract and validate scores
        bm25_scores = [doc.get('bm25_score', 0) for doc in documents]
        vector_scores = [doc.get('vector_score', 0) for doc in documents]
        print(bm25_scores)
        
        # Normalize scores
        norm_bm25 = self._normalize(bm25_scores)
        norm_vector = self._normalize(vector_scores)
        
        # Combine scores
        for doc, bm25, vector in zip(documents, norm_bm25, norm_vector):
            doc['combined_score'] = (self.bm25_weight * bm25) + (self.vector_weight * vector)
            # Keep original scores for reference
            doc['normalized_bm25'] = bm25
            doc['normalized_vector'] = vector
            
        return sorted(documents, key=lambda x: x['combined_score'], reverse=True)