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

    # Not support expansion yet
    def bm25_only_search(self, query, top_n):
        print("bm25 only search")
        # BM25 
        check_elasticsearch_server()
        if len(query) <= 1:
            # BM25-only search (no vector similarity)
            search_query = {
                "query": {
                    "match": {
                        "prepared_text": query[0]
                    }
                },
                "size": top_n
            }
            response = self.es.search(index="arxiv_index", body=search_query)
        else:
            search_query = []
            original_query = query[0]
            for term in query:
                search_query.append({"index": "arxiv_index"})
                search_query.append({
                "query": {
                    "match": {
                        "prepared_text": {
                            "query": term,
                            "boost": 2.0 if term == original_query else 1.0
                        }
                    }
                },
                "size": top_n
            })
            response = self.es.msearch(body=search_query)
        return response

    def bert_only_search(self, query, top_n=8):
        print("Bert only search")
        check_elasticsearch_server()

        if len(query) <= 1:
            query_embedding = self.model.encode(query[0]).tolist()
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
                        "score_mode": "max"
                    }
                },
                "size": top_n
            }
            response = self.es.search(body=search_query)
        else:
            search_query = []
            original_query = query[0]
            for term in query:
                boost = 2.0 if term == original_query else 1.0
                term_embedding = self.model.encode(term).tolist()
                # Add to multi-search body
                search_query.append({"index": "arxiv_index"})
                search_query.append({
                    "query": {
                        "nested": {
                            "path": "paragraphs",
                            "query": {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'paragraphs.embedding') * params.boost + 1.0",
                                        "params": {
                                            "query_vector": term_embedding,
                                            "boost": boost
                                        }
                                    }
                                }
                            },
                            "score_mode": "max"
                        }
                    },
                    "size": top_n
                })

            response = self.es.msearch(body=search_query)

        return response

    def hybrid_search(self, query, top_n, mode="max"):
        print("Hybrid search with weighted query terms")
        check_elasticsearch_server()
        
        # Calculate weights
        original_weight = 0.6  # 60% for original query
        expansion_weight = 0.4  # 40% for all expansion terms combined
        
        if len(query) <= 1:
            # Single query case (no expansion)
            query = query[0]
            query_embedding = self.model.encode(query).tolist()
            search_query = [
                {"index": "arxiv_index"},
                {
                    "query": {"match": {"prepared_text": query}},
                    "size": top_n,
                    "explain": True
                },
                {"index": "arxiv_index"},
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
                            "score_mode": mode
                        }
                    },
                    "size": top_n,
                    "explain": True
                }
            ]
        else:
            # Query expansion case
            original_query = query[0]
            expansion_terms = query[1:]
            # print(f"query: {query}")
            # print(f"origin: {original_query}")
            # print(f"expanded: {expansion_terms}")
            num_expansion_terms = len(expansion_terms) if expansion_terms else 1
            
            search_query = []
            for term in query:
                term_embedding = self.model.encode(term).tolist()
                
                # Calculate term-specific weight
                if term == original_query:
                    term_weight = original_weight  # 60% for original
                else:
                    term_weight = expansion_weight / num_expansion_terms  # Distribute 40% among expansions
                
                # BM25 search with weight
                search_query.append({"index": "arxiv_index"})
                search_query.append({
                    "query": {
                        "match": {
                            "prepared_text": {
                                "query": term,
                                "boost": term_weight * 2.0  # Scale for Elasticsearch boost
                            }
                        }
                    },
                    "size": top_n,
                    "explain": True
                })

                # Vector search with weight
                search_query.append({"index": "arxiv_index"})
                search_query.append({
                    "query": {
                        "nested": {
                            "path": "paragraphs",
                            "query": {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": f"""
                                        cosineSimilarity(params.query_vector, 'paragraphs.embedding') 
                                        * {term_weight} 
                                        + 1.0
                                        """,
                                        "params": {"query_vector": term_embedding}
                                    }
                                }
                            },
                            "score_mode": mode
                        }
                    },
                    "size": top_n,
                    "explain": True
                })

        # Execute the search
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
            results = []
            num_terms = len(query) if len(query) > 1 else 1

            if len(query) <= 1:
                bm25_res = response["responses"][0]["hits"]["hits"]
                vector_res = response["responses"][1]["hits"]["hits"]
                # Extract results
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
                for i in range(0, len(response["responses"]), 2):  # Step by 2 for BM25+vector pairs
                    bm25_res = response["responses"][i]["hits"]["hits"]
                    vector_res = response["responses"][i + 1]["hits"]["hits"]
                    # Take top_n results for this term
                    for bm25, bert in zip(bm25_res, vector_res):
                        bm25_score = bm25["_score"] 
                        vector_score = bert["_score"] 
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
            if "responses" in response:  # msearch case
                print("Handling msearch case")
                for res in response["responses"]:
                    for hit in res["hits"]["hits"]:
                        results.append({
                            "id": hit["_id"],
                            "title": hit["_source"]["title"],
                            "abstract": hit["_source"]["abstract"],
                            "score": hit["_score"]
                        })

            else:  # single search case
                for hit in response["hits"]["hits"]:
                    results.append({
                        "id": hit["_id"],
                        "title": hit["_source"]["title"],
                        "abstract": hit["_source"]["abstract"],
                        "score": hit["_score"]
                    })

        return results
