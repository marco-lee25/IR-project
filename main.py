
from elasticsearch import Elasticsearch
from database.process_data import build_index_system
import search_engine
import json
import argparse
from preprocess import preprocess_sys
from summarizer_utils import BartSummarizer, summarize_text

def process_input(se, query, use_bm25=True, use_bert=False, top_n=5, summarizer=None):
    print(f"Query: {query}")
    print(f"Use BM25: {use_bm25}")
    print(f"Use BERT: {use_bert}")

    # Call your search function with parameters
    results = se.search(query, use_bm25=use_bm25, use_bert=use_bert, top_n=top_n)
    print("===================================")
    print(len(results))
    # Print the search results
    
    for doc in results:
        if not summarizer is None:
            summary = summarizer.summarize(doc["abstract"])
            summary2=summarize_text(doc["abstract"], query)
            doc["summary"] = summary
            doc["summary2"] = summary2

            if 'bm25_score' in doc and "vector_score" in doc:
                print(f"Title: {doc['title']}\n Abstract: {doc['abstract']}\n Summary: {summary}\n bm25_score:{doc['bm25_score']}\n vector_score:{doc['vector_score']}\n")
                # print("********************","Summary2: {summary2}\n")
                print(f"******************** Summary2: {summary2}\n")

            else:
                print(f"Title: {doc['title']}\n Abstract: {doc['abstract']}\n Summary: {summary}\n Summary2: {summary2}\n score:{doc['score']}\n")
                print(f"******************** Summary2: {summary2}\n")

        else:
            # summary2=summarize_text(doc["abstract"], query)
            # doc["summary2"] = summary2
            print("no summarizer")
            if 'bm25_score' in doc and "vector_score" in doc:
                print(f"Title: {doc['title']}\n Abstract: {doc['abstract']}\n bm25_score:{doc['bm25_score']}\n vector_score:{doc['vector_score']}\n")

            else:
                print(f"Title: {doc['title']}\n Abstract: {doc['abstract']}\n score:{doc['score']}\n")
            # print(f"******************** Summary2: {summary2}\n")
            # print(f"!!!!!1 Summary2: {doc['summary2']}\n")


# Example usage
# No expansion
# python main.py "face identify" --use_bm25 --use_bert
# With expansion on synoyms
# python main.py "face identify" --use_bm25 --use_bert --use_expansion --exp_syn
if __name__ == "__main__":
    preprocess = preprocess_sys()

    print("Initalizing search engine...")
    se = search_engine.engine()

    parser = argparse.ArgumentParser(description="Run the search engine with parameters.")

    # Positional argument: Query
    parser.add_argument("query", type=str, help="Search query")

    # Optional flags
    parser.add_argument("--use_bm25", action="store_true", help="Enable BM25-based search")
    parser.add_argument("--use_bert", action="store_true", help="Enable BERT-based semantic search")
    parser.add_argument("--use_expansion", action="store_true", help="Query expansion")
    parser.add_argument("--exp_syn", action="store_true", help="Apply synoyms expansion")
    parser.add_argument("--exp_sem", action="store_true", help="Query semantic expansion")
    parser.add_argument("--top_n", type=int,default=5, help='Max number of documents return')
    parser.add_argument("--use_summary", action="store_true", help='Enable BART summarization')
    
    args = parser.parse_args()

    summarizer = BartSummarizer() if args.use_summary else None 

    if args.use_expansion:
        if not(args.exp_syn) and not(args.exp_sem):
            print("Please specify expansion method by --exp_syn & --exp_sem")
            exit()
        processed_query = preprocess.process_query(args.query,  use_semantic=args.exp_sem, use_synonyms=args.exp_syn)
        print(f"Query expansion result : {processed_query}")

        # TODO
        # Handle the expaned query, for example combining into single string or separate to different query to search.
        # processed_query = ' '.join(processed_query)
        # print(processed_query)
        # exit()
        
        # Run main function with parsed arguments
        process_input(se, processed_query, args.use_bm25, args.use_bert, args.top_n, summarizer=summarizer)
    else:
        process_input(se, [args.query], args.use_bm25, args.use_bert, args.top_n, summarizer=summarizer)
