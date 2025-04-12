from elasticsearch import Elasticsearch
from database.process_data import build_index_system
import search_engine
import json
import argparse
from preprocess_system.preprocess import preprocess_sys
from summarize_system.summarizer import BartSummarizer
from ranking_system.ranking_function import HybridRanker
import torch
from models.model import scibert_model

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
            if use_bm25 == True and use_bert ==False:
                output.append(f"BM25: {doc['score']:.3f}")
            if use_bm25 == False and use_bert ==True:
                output.append(f"Vector: {doc['score']:.3f}")
            
        if summarizer:
            output.append(f"Summary: {summarizer.summarize(doc['abstract'])}")
            
        print("\n".join(output) + "\n" + "-"*50)
# =================================================

def compare_rankings(results_hybrid, results_bm25, results_bert, ranker):
    """Show difference between all three ranking methods"""
    # Create independent copies for each ranking method
    bm25_only = [doc.copy() for doc in results_bm25]
    vector_only = [doc.copy() for doc in results_bert]
    hybrid = [doc.copy() for doc in results_hybrid]

    # Apply each ranking method separately
    bm25_sorted = sorted(bm25_only, key=lambda x: x['score'], reverse=True)
    vector_sorted = sorted(vector_only, key=lambda x: x['score'], reverse=True)
    hybrid_sorted = ranker.rank_documents(results_bm25, results_bert)

    print("\n=== RANKING COMPARISON ===")
    print(f"{'BM25 Order':<40} | {'Vector Order':<40} | {'Hybrid Order':<40}")
    print("-" * 120)
    
    for i in range(min(5, len(bm25_sorted))):
        bm25_title = bm25_sorted[i]['title'][:35] + (bm25_sorted[i]['title'][35:] and '...')
        vector_title = vector_sorted[i]['title'][:35] + (vector_sorted[i]['title'][35:] and '...')
        hybrid_title = hybrid_sorted[i]['title'][:35] + (hybrid_sorted[i]['title'][35:] and '...')
        
        print(f"{bm25_title:<40} | {vector_title:<40} | {hybrid_title:<40} ")
        print(f"BM25: {bm25_sorted[i]['score']:.2f} | "
              f"Vector: {vector_sorted[i]['score']:.2f} | "
              f"Combined: {hybrid_sorted[i].get('combined_score', 0):.2f}"
            )
        print("-" * 120)

def process_input_compare_ranking(se, query, use_bm25=True, use_bert=True, top_n=5, summarizer=None):
    print(f"Query: {query}")
    print(f"BM25: {use_bm25}, Vector: {use_bert}")
    
    # Run search and get raw results
    results_hybrid = se.search(query, use_bm25=use_bm25, use_bert=use_bert, top_n=top_n)
    results_bm25 = se.search(query, use_bm25=use_bm25, use_bert=False, top_n=top_n)
    results_bert = se.search(query, use_bm25=False, use_bert=use_bert, top_n=top_n)
    
    # Only compare if both scores are available
    if use_bm25 and use_bert:
        ranker = HybridRanker(bm25_weight=args.bm25_weight, vector_weight=args.vector_weight)
        
        # Get final hybrid ranked results
        ranked_results = ranker.rank_documents(results_bm25, results_bert)
        compare_rankings(ranked_results, results_bm25, results_bert, ranker)
        # print(f"ranked results: {ranked_results}")
        display_results(ranked_results, top_n, summarizer)
    else:
        display_results(results_hybrid, top_n, summarizer)

def display_results(results, top_n, summarizer, ranking_method="Hybrid"):
    """Display formatted results with ranking method context"""
    print(f"\n=== {ranking_method.upper()} RANKING RESULTS ===")
    
    for i, doc in enumerate(results[:top_n], 1):
        # Base output with title and abstract
        output = [
            f"Rank {i}: {doc['title']}",
            f"Abstract: {doc['abstract'][:150]}{'...' if len(doc['abstract']) > 150 else ''}"
        ]
        
        # Score information
        score_info = []
        if 'bm25_score' in doc:
            score_info.append(f"BM25: {doc['bm25_score']:.2f}")
        if 'vector_score' in doc:
            score_info.append(f"Vector: {doc['vector_score']:.2f}")
        if 'combined_score' in doc:
            score_info.append(f"Combined: {doc['combined_score']:.2f}")
            if 'normalized_bm25' in doc and 'normalized_vector' in doc:
                score_info.append(
                    f"(Norm: BM25={doc['normalized_bm25']:.2f}, "
                    f"Vector={doc['normalized_vector']:.2f})"
                )
        
        if score_info:
            output.append("Scores: " + " | ".join(score_info))
        
        # Summary if enabled
        if summarizer:
            output.append(f"Summary: {summarizer.summarize(doc['abstract'])}")
        
        print("\n".join(output))
        print("=" * 80)

# =================================================

def process_input_no_rank(se, query, use_bm25=True, use_bert=False, top_n=5, summarizer=None):
    print(f"Query: {query}")
    print(f"BM25: {use_bm25}, Vector: {use_bert}")
    
    results = se.search(query, use_bm25=use_bm25, use_bert=use_bert, top_n=top_n)
    
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
            if use_bm25 == True and use_bert ==False:
                output.append(f"BM25: {doc['score']:.3f}")
            if use_bm25 == False and use_bert ==True:
                output.append(f"Vector: {doc['score']:.3f}")
            
        if summarizer:
            output.append(f"Summary: {summarizer.summarize(doc['abstract'])}")
            
        print("\n".join(output) + "\n" + "-"*50)

# Example usage 
# No expansion
# python main.py "face identify" --use_bm25 --use_bert
# With expansion on synoyms
# python main.py "face identify" --use_bm25 --use_bert --use_expansion --exp_syn
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = scibert_model(device)

    print("Initalizing preprocess system...")
    preprocess = preprocess_sys(model, device)

    print("Initalizing search engine...")
    se = search_engine.engine(model)

    parser = argparse.ArgumentParser(description="Run the search engine with parameters.")

    # Positional argument: Query
    parser.add_argument("query", type=str, help="Search query") 

    # Optional flags
    parser.add_argument("--use_bm25", action="store_true", help="Enable BM25-based search")
    parser.add_argument("--use_bert", action="store_true", help="Enable BERT-based semantic search")
    parser.add_argument("--use_expansion", action="store_true", help="Query expansion")
    parser.add_argument("--exp_syn", action="store_true", help="Apply synoyms expansion")
    parser.add_argument("--exp_sem", action="store_true", help="Query semantic expansion")
    parser.add_argument("--top_n", type=int,default=10, help='Max number of documents return')
    parser.add_argument("--use_summary", action="store_true", help='Enable BART summarization')
    parser.add_argument("--bm25_weight", type=float, default=0.5, 
                   help="Weight for BM25 in hybrid ranking (0.0-1.0)")
    parser.add_argument("--vector_weight", type=float, default=0.5,
                   help="Weight for vector search in hybrid ranking (0.0-1.0)")
    parser.add_argument("--sem_method", type=int, default=1,
               help=" 0:Semantic expansion on GoogleNews-vectors\n 1: Expansion using GenAI")
    
    args = parser.parse_args()

    summarizer = BartSummarizer(device) if args.use_summary else None 

    if args.use_expansion:
        if not(args.exp_syn) and not(args.exp_sem):
            print("Please specify expansion method by --exp_syn & --exp_sem")
            exit()
        processed_query = preprocess.process_query(args.query,  use_semantic=args.exp_sem, use_synonyms=args.exp_syn, sem_method=args.sem_method)
        print(f"Query expansion result : {processed_query}")

        # TODO 
        # Handle the expaned query, for example combining into single string or separate to different query to search.
        # processed_query = ' '.join(processed_query)
        # print(processed_query)
        # exit()
        
        # Run main function with parsed arguments
    #     process_input(se, processed_query, args.use_bm25, args.use_bert, args.top_n, summarizer=summarizer)
    # else:
    #     process_input(se, [args.query], args.use_bm25, args.use_bert, args.top_n, summarizer=summarizer)

        process_input_compare_ranking(se, processed_query, args.use_bm25, args.use_bert, args.top_n, summarizer=summarizer)
    else:
        process_input_compare_ranking(se, [args.query], args.use_bm25, args.use_bert, args.top_n, summarizer=summarizer)