import numpy as np
from typing import List, Dict

class HybridRanker:
    def __init__(self, bm25_weight: float = 0.5, vector_weight: float = 0.5):
        self.bm25_weight = bm25_weight
        self.bert_weight = vector_weight
        
        # Validate weights sum to 1
        if not np.isclose(bm25_weight + vector_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")

    def _sigmoid_normalize(self, scores: List[float]) -> List[float]:
        """More robust normalization for score distributions"""
        x = np.array(scores)
        if len(x) == 0:
            return []
        if np.std(x) < 1e-6:  # All scores similar
            return np.ones_like(x).tolist()
            
        # Scale to [0,1] using sigmoid
        x_normalized = 1 / (1 + np.exp(-(x - np.mean(x)) / np.std(x)))
        return (x_normalized / np.max(x_normalized)).tolist()

    def _scale_scores(self, bm25_scores: List[float], 
                     bert_scores: List[float]) -> tuple:
        """Scale scores to comparable ranges"""
        # BM25 typically has larger magnitude
        bm25_scaled = np.array(bm25_scores) / 20  # Empirical scaling factor
        bert_scaled = (np.array(bert_scores) + 1) / 2  # [-1,1] → [0,1]
        return bm25_scaled.tolist(), bert_scaled.tolist()

    def rank_documents(self, bm25_results: List[Dict], bert_results: List[Dict]) -> List[Dict]:
        """
        Rank documents by combining BM25 and BERT scores from separate result sets.
        
        Args:
            bm25_results: List of documents with BM25 scores (e.g., {'doc_id': str, 'bm25_score': float, ...})
            bert_results: List of documents with BERT scores (e.g., {'doc_id': str, 'vector_score': float, ...})
        
        Returns:
            List of documents sorted by combined score, with normalized scores included
        """

        # Create a unified list of documents by merging on doc_id
        doc_map = {}
        
        # Process BM25 results
        for doc in bm25_results:
            doc_id = doc.get('id')
            if doc_id is None:
                continue  # Skip documents without an ID
            doc_map[doc_id] = doc_map.get(doc_id, {'doc_id': doc_id})
            doc_map[doc_id]['bm25_score'] = doc.get('score', 0)
            doc_map[doc_id].update({k: v for k, v in doc.items() if k not in ['bm25_score', 'vector_score', 'combined_score']})


        # Process BERT results
        for doc in bert_results:
            doc_id = doc.get('id')
            if doc_id is None:
                continue  # Skip documents without an ID
            doc_map[doc_id] = doc_map.get(doc_id, {'doc_id': doc_id})
            doc_map[doc_id]['vector_score'] = doc.get('score', 0)
            doc_map[doc_id].update({k: v for k, v in doc.items() if k not in ['bm25_score', 'vector_score', 'combined_score']})

        # Convert to list and handle missing scores
        merged_docs = list(doc_map.values())
        bm25_scores = [doc.get('bm25_score', 0) for doc in merged_docs]
        bert_scores = [doc.get('vector_score', 0) for doc in merged_docs]

        # Scale and normalize scores
        bm25_scaled, bert_scaled = self._scale_scores(bm25_scores, bert_scores)
        norm_bm25 = self._sigmoid_normalize(bm25_scaled)
        norm_bert = self._sigmoid_normalize(bert_scaled)

        # Combine scores and store normalized values
        for doc, bm25, bert in zip(merged_docs, norm_bm25, norm_bert):
            doc['combined_score'] = (
                self.bm25_weight * bm25 + 
                self.bert_weight * bert
            )
            doc['normalized_bm25'] = bm25
            doc['normalized_vector'] = bert

        # Sort by combined score
        return sorted(merged_docs, key=lambda x: x['combined_score'], reverse=True)