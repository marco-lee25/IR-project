from search_engine import search
from elasticsearch import Elasticsearch
from database.process_data import build_index_system
import search_engine
import json
import argparse


def process_input(query, use_bm25=True, use_bert=False, top_n=5):
    print(f"Query: {query}")
    print(f"Use BM25: {use_bm25}")
    print(f"Use BERT: {use_bert}")

    # Call your search function with parameters
    results = search(query, use_bm25=use_bm25, use_bert=use_bert, top_n=top_n)

    # Print the search results
    for doc in results:
        if 'bm25_score' in doc and "vector_score" in doc:
            print(f"Title: {doc['title']}\n Abstract: {doc['abstract']}\n bm25_score:{doc['bm25_score']}\n vector_score:{doc['vector_score']}\n")
        else:
            print(f"Title: {doc['title']}\n Abstract: {doc['abstract']}\n score:{doc['score']}\n")


# python main.py "face identify" --use_bm25 True --use_bert True
if __name__ == "__main__":
    # build_index_system(index_name="arxiv_index", use_bert=True, max_doc=1000)

    # es = Elasticsearch("http://localhost:9200")
    # # Get the index mapping
    # mapping = es.indices.get_mapping(index="arxiv_index")

    # # Pretty print the JSON response
    # print(json.dumps(mapping, indent=4))

    parser = argparse.ArgumentParser(description="Run the search engine with parameters.")

    # Positional argument: Query
    parser.add_argument("query", type=str, help="Search query")

    # Optional flags
    parser.add_argument("--use_bm25", type=bool, default=True, help="Enable BM25-based search")
    parser.add_argument("--use_bert", type=bool, default=False, help="Enable BERT-based semantic search")
    parser.add_argument("--top_n", type=int,default=5, help='Max number of documents return')

    args = parser.parse_args()
    
    # Run main function with parsed arguments
    process_input(args.query, args.use_bm25, args.use_bert, args.top_n)
