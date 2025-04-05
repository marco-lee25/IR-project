from elasticsearch import Elasticsearch
from database.indexing_system import check_elasticsearch_server
from sentence_transformers import models, SentenceTransformer

#TODO 
# Scoring may need to edit the srcipt score ot function score to combine BM25 and Cosine Similarity to give a better result.


class engine():
    def __init__(self):
        self.es = Elasticsearch("http://localhost:9200")  # Connect to Elasticsearch
        word_embedding_model = models.Transformer(
            'allenai/scibert_scivocab_uncased',
            max_seq_length=128,
            do_lower_case=True
        )
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    def bm25_only_search(self, query, top_n):
        print("bm25 only search")
        # BM25 
        check_elasticsearch_server()

        # BM25-only search (no vector similarity)
        search_query = {
            "query": {
                "match": {
                    "prepared_text": query
                }
            },
            "size": top_n
        }
        response = self.es.search(index="arxiv_index", body=search_query)
        return response

    def bert_only_search(self, query, top_n=8):
        print("Bert only search")
        check_elasticsearch_server()
        query_embedding = self.model.encode(query).tolist()
        search_query = {
            "query": {
                "nested": {
                    "path": "paragraphs",
                    "query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'paragraphs.embedding') + 1.0",
                                "params": {"query_vector": query_embedding}
                            }
                        }
                    },
                    "score_mode": "max"  # Use the highest paragraph similarity score
                }
            },
            "size": top_n
        }
        response = self.es.search(index="arxiv_index", body=search_query)
        return response

    def hybrid_search(self, query, top_n, mode="max"):
        print("Hybrid search")
        # BM25 + Bert
        check_elasticsearch_server()
        query_embedding = self.model.encode(query).tolist()
        # Generated using GPT, search query with separate BM25 and vector scoring
        search_query = [
            # -- Query 1: BM25
            {"index": "arxiv_index"},
            {
                "query": {"match": {"prepared_text": query}},
                "size": top_n,
                "explain": True
            },

            # -- Query 2: Script Score (Vector)
            {"index": "arxiv_index"},  # Updated index name
            {
                "query": {
                    "nested": {
                        "path": "paragraphs",
                        "query": {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'paragraphs.embedding') + 1.0",
                                    "params": {"query_vector": query_embedding}
                                }
                            }
                        },
                        "score_mode": mode  # Use the highest paragraph similarity score
                    }
                },
                "size": top_n,
                "explain": True
            }
        ]
        response = self.es.msearch(body=search_query)
        return response

    def search(self, query, use_bm25=True, use_bert=True, top_n=5, alpha=0.5):
        if use_bert and use_bm25:
            response = self.hybrid_search(query, top_n)
        elif use_bert:
            response = self.bert_only_search(query, top_n)
        else :
            response = self.bm25_only_search(query, top_n)

        if use_bert and use_bm25:
            bm25_res = response["responses"][0]["hits"]["hits"]
            vector_res = response["responses"][1]["hits"]["hits"]
            # Extract results
            results = []
            for bm25, bert in zip(bm25_res, vector_res):
                bm25_score = bm25["_score"]  # BM25 relevance score
                vector_score = bert["_score"]  # Dense vector score (if exists)

                # # Combine BM25 and vector search scores using weighted sum
                combined_score = alpha * vector_score + (1 - alpha) * bm25_score

                results.append({
                    "id": bm25["_id"],
                    "title": bm25["_source"]["title"],
                    "abstract": bm25["_source"]["abstract"],
                    "bm25_score": bm25_score,
                    "vector_score": vector_score
                })

        else:
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "id": hit["_id"],
                    "title": hit["_source"]["title"],
                    "abstract": hit["_source"]["abstract"],
                    "score": hit["_score"]  # Relevance score
                })
        return results
