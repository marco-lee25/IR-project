from search_engine import search_elasticsearch
from elasticsearch import Elasticsearch
from database.process_data import build_index_system
import search_engine
import json

if __name__ == "__main__":
    # build_index_system(index_name="arxiv_index", use_bert=True, max_doc=100)
    # es = Elasticsearch("http://localhost:9200")

    # # Get the index mapping
    # mapping = es.indices.get_mapping(index="arxiv_index")

    # # Pretty print the JSON response
    # print(json.dumps(mapping, indent=4))

    query = "face identify"
    results = search_engine.search(query, use_bm25=True, use_bert=True)
    
    for doc in results:
        if 'bm25_score' in doc and "vector_score" in doc:
            print(f"Title: {doc['title']}\n Abstract: {doc['abstract']}\n bm25_score:{doc['bm25_score']}\n vector_score:{doc['vector_score']}\n")
        else:
            print(f"Title: {doc['title']}\n Abstract: {doc['abstract']}\n score:{doc['score']}\n")
