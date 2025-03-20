from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from database.indexing_system import check_elasticsearch_server

def old_search_elasticsearch(query, top_n=5, use_bert=True, alpha=0.5):
    # BM25 + Bert
    es = Elasticsearch("http://localhost:9200")  # Connect to Elasticsearch
    check_elasticsearch_server()

    if use_bert:
        model = SentenceTransformer("allenai/scibert_scivocab_uncased")
        query_embedding = model.encode(query).tolist()

        # Ensure embedding has 768 dimensions
        if len(query_embedding) != 768:
            raise ValueError(f"Query embedding has {len(query_embedding)} dimensions, expected 768")
        
        # Generated using GPT, search query with separate BM25 and vector scoring

        search_query = {
            "size": top_n,
            "query": {
                "bool": {
                    "filter": {"exists": {"field": "embedding"}},  # Only docs with embeddings
                    "must": {"match": {"prepared_text": query}}  # BM25 scoring
                }
            },
            "script_fields": {
                "vector_score": {
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            }
        }

        # search_query = {
        #     "size": top_n,
        #     "query": {
        #         "match": {"prepared_text": query}  # BM25 base query
        #     },
        #     "script_fields": {
        #         "vector_score": {
        #             "script": {
        #                 "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
        #                 "params": {"query_vector": query_embedding}
        #             }
        #         }
        #     }
        # }


        search_query = {
            "size": top_n,
            "explain": True,
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "prepared_text": query
                            }
                        },
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {"query_vector": query_embedding}
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }
    else:
        # BM25-only search (no vector similarity)
        search_query = {
            "query": {
                "match": {
                    "prepared_text": query
                }
            },
            "size": top_n
        }

    response = es.search(index="arxiv_index", body=search_query)
    
    if use_bert:
        # Extract results
        results = []
        for hit in response["hits"]["hits"]:
            combined_score = hit["_score"]  # BM25 relevance score
            # vector_score = hit["fields"]["vector_score"][0]  # Dense vector score (if exists)

            # # Combine BM25 and vector search scores using weighted sum
            # combined_score = alpha * vector_score + (1 - alpha) * bm25_score

            results.append({
                "id": hit["_id"],
                "title": hit["_source"]["title"],
                "abstract": hit["_source"]["abstract"],
                # "bm25_score": bm25_score,
                # "vector_score": vector_score,
                "final_score": combined_score  # Hybrid score
            })

        # Sort results by combined hybrid score
        results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        return results
    else: 
        # Extract results
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "title": hit["_source"]["title"],
                "abstract": hit["_source"]["abstract"],
                "final_score": hit["_score"]  # Relevance score
            })

        return results
    

def bm25_only_search(query, top_n):
    print("bm25 only search")
    # BM25 
    es = Elasticsearch("http://localhost:9200")  # Connect to Elasticsearch
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
    response = es.search(index="arxiv_index", body=search_query)
    return response

def bert_only_search(query, top_n):
    print("Bert only search")
    es = Elasticsearch("http://localhost:9200")  # Connect to Elasticsearch
    check_elasticsearch_server()
    model = SentenceTransformer("allenai/scibert_scivocab_uncased")
    query_embedding = model.encode(query).tolist()
    search_query = {
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        },
        "size": top_n
    }
    response = es.search(index="arxiv_index", body=search_query)
    return response

def hybrid_search(query, top_n):
    print("Hybrid search")
    # BM25 + Bert
    es = Elasticsearch("http://localhost:9200")  # Connect to Elasticsearch
    check_elasticsearch_server()
    model = SentenceTransformer("allenai/scibert_scivocab_uncased")
    query_embedding = model.encode(query).tolist()
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
        {"index": "arxiv_index"},
        {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            },
            "size": top_n,
            "explain": True
        }
    ]
    response = es.msearch(body=search_query)
    return response

def search(query, use_bm25=True, use_bert=True, top_n=5, alpha=0.5):
    if use_bert and use_bm25:
        response = hybrid_search(query, top_n)
    elif use_bert:
        response = bert_only_search(query, top_n)
    else :
        response = bm25_only_search(query, top_n)

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

def search_elasticsearch(query, top_n=5, use_bm25=True, use_bert=False, alpha=0.5):
    # BM25 + Bert
    es = Elasticsearch("http://localhost:9200")  # Connect to Elasticsearch
    check_elasticsearch_server()

    if use_bert:
        model = SentenceTransformer("allenai/scibert_scivocab_uncased")
        query_embedding = model.encode(query).tolist()

        # Ensure embedding has 768 dimensions
        if len(query_embedding) != 768:
            raise ValueError(f"Query embedding has {len(query_embedding)} dimensions, expected 768")
        
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
            {"index": "arxiv_index"},
            {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "size": top_n,
                "explain": True
            }
        ]
        response = es.msearch(body=search_query)
    else:
        # BM25-only search (no vector similarity)
        search_query = {
            "query": {
                "match": {
                    "prepared_text": query
                }
            },
            "size": top_n
        }
        response = es.search(index="arxiv_index", body=search_query)

    if use_bert:
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
                "vector_score": vector_score,
                "final_score": combined_score  # Hybrid score
            })

        # Sort results by combined hybrid score
        results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        return results
    else: 
        # Extract results
        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "id": hit["_id"],
                "title": hit["_source"]["title"],
                "abstract": hit["_source"]["abstract"],
                "final_score": hit["_score"]  # Relevance score
            })

        return results