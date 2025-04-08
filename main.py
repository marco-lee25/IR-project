
from elasticsearch import Elasticsearch
from database.process_data import build_index_system
import search_engine
import json
import argparse
from preprocess_system.preprocess import preprocess_sys
from summarize_system.summarizer import BartSummarizer
from ranking_system.ranking_function import HybridRanker

def process_input(se, query, use_bm25=True, use_bert=False, top_n=5, summarizer=None, ranker=None):
    print(f"Query: {query}")
    print(f"BM25: {use_bm25}, Vector: {use_bert}")
    
    results = se.search(query, use_bm25=use_bm25, use_bert=use_bert, top_n=top_n)
    ranker = HybridRanker(bm25_weight=args.bm25_weight, vector_weight=args.vector_weight)
    # Apply hybrid ranking if both scores exist
    if all('bm25_score' in doc and 'vector_score' in doc for doc in results) and ranker:
        results = ranker.rank_documents(results)
    
    print("="*50)
    for i, doc in enumerate(results[:top_n], 1):
        output = [
            f"RESULT {i}:",
            f"Title: {doc['title']}",
            f"Abstract: {doc['abstract'][:200]}...",
        ]
        
        if 'combined_score' in doc:
            output.extend([
                f"BM25: {doc['bm25_score']:.3f} (norm: {doc['normalized_bm25']:.3f})",
                f"Vector: {doc['vector_score']:.3f} (norm: {doc['normalized_vector']:.3f})",
                f"Combined: {doc['combined_score']:.3f}"
            ])
        else:
            output.append(f"BM25: {doc['bm25_score']:.3f}\n Vector: {doc['vector_score']:.3f}")
            
        if summarizer:
            output.append(f"Summary: {summarizer.summarize(doc['abstract'])}")
            
        print("\n".join(output) + "\n" + "-"*50)


def process_input_no_rank(se, query, use_bm25=True, use_bert=False, top_n=5, summarizer=None):
    print(f"Query: {query}")
    print(f"BM25: {use_bm25}, Vector: {use_bert}")
    
    results = se.search(query, use_bm25=use_bm25, use_bert=use_bert, top_n=top_n)
    ranker = HybridRanker(bm25_weight=args.bm25_weight, vector_weight=args.vector_weight)
    # Apply hybrid ranking if both scores exist
    if all('bm25_score' in doc and 'vector_score' in doc for doc in results) and ranker:
        results = ranker.rank_documents(results)
    
    print("="*50)
    for i, doc in enumerate(results[:top_n], 1):
        output = [
            f"RESULT {i}:",
            f"Title: {doc['title']}",
            f"Abstract: {doc['abstract'][:200]}...",
        ]
        
        if 'combined_score' in doc:
            output.extend([
                f"BM25: {doc['bm25_score']:.3f} (norm: {doc['normalized_bm25']:.3f})",
                f"Vector: {doc['vector_score']:.3f} (norm: {doc['normalized_vector']:.3f})",
                f"Combined: {doc['combined_score']:.3f}"
            ])
        else:
            output.append(f"BM25: {doc['bm25_score']:.3f}\n Vector: {doc['vector_score']:.3f}")
            
        if summarizer:
            output.append(f"Summary: {summarizer.summarize(doc['abstract'])}")
            
        print("\n".join(output) + "\n" + "-"*50)
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
    parser.add_argument("--bm25_weight", type=float, default=0.5, 
                   help="Weight for BM25 in hybrid ranking (0.0-1.0)")
    parser.add_argument("--vector_weight", type=float, default=0.5,
                   help="Weight for vector search in hybrid ranking (0.0-1.0)")
    
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
