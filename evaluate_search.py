from elasticsearch import Elasticsearch
import search_engine
import json
import argparse
from preprocess_system.preprocess import preprocess_sys
from summarize_system.summarizer import BartSummarizer
from ranking_system.ranking_function import HybridRanker
import torch
from models.model import scibert_model, deepseek_model
import pytrec_eval
import os

def save_trec_run(results, query_id, run_name, output_file="results.trec"):
    with open(output_file, 'a') as f:
        for rank, doc in enumerate(results, 1):
            score = doc.get('combined_score', doc.get('score', 0.0))
            doc_id = doc.get('id', 'UNKNOWN')
            f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_name}\n")

def evaluate_trec(qrels_file, run_file, verbose=False):
    qrels = {}
    with open(qrels_file, 'r') as f_qrels:
        for i, line in enumerate(f_qrels, 1):
            line = line.strip()
            # Hardcode skipping the first line
            if i == 1:
                print(f"Skipping first line: {line}")
                continue
            parts = line.split()
            qid, _, docid, rel = parts
            if not qid.isdigit():
                print(f"Warning: Skipping line {i} with invalid qid: {line}")
                continue
            try:
                rel_val = int(rel)
            except ValueError:
                print(f"Warning: Skipping line {i} with invalid rel: {line}")
                continue
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = rel_val

    if not qrels:
        raise ValueError("No valid qrels data loaded. Check file format.")

    run = {}
    with open(run_file, 'r') as f_run:
        for i, line in enumerate(f_run, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                print(f"Warning: Skipping malformed run line {i}: {line}")
                continue
            qid, _, docid, _, score, _ = parts
            if qid not in run:
                run[qid] = {}
            try:
                run[qid][docid] = float(score)
            except ValueError:
                print(f"Warning: Skipping run line with invalid score: {line}")
                continue

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {'map', 'ndcg', 'P_5', 'P_10', 'recall_5', 'recall_10'}
    )
    results = evaluator.evaluate(run)

    if not results:
        print("No evaluation results generated.")
        return {}, {}

    # Calculate aggregated metrics
    aggregated = {}
    metrics = ['map', 'ndcg', 'P_5', 'P_10', 'recall_5', 'recall_10']
    for metric in metrics:
        valid_results = [results[qid][metric] for qid in results if metric in results[qid]]
        aggregated[metric] = sum(valid_results) / len(valid_results) if valid_results else 0.0

    print(f"\nAveraged metrics over {len(results)} queries:")
    for metric, value in aggregated.items():
        print(f"{metric}: {value:.4f}")

    if verbose:
        print("\nPer-Query Metrics:")
        for qid in results:
            print(f"\nQuery ID: {qid}")
            for metric, value in results[qid].items():
                print(f"  {metric}: {value:.4f}")

    return aggregated, results

def load_queries(query_file, max_queries=100):
    with open(query_file, 'r') as f:
        queries = json.load(f)
    return queries[:max_queries]  # Limit to first 100 queries

def process_queries(se, queries, use_bm25=True, use_bert=False, top_n=5, summarizer=None, bm25_weight=0.7, vector_weight=0.3, qrels_file="test_set_qrels.txt", run_file="results.trec", verbose=False):
    if os.path.exists(run_file):
        os.remove(run_file)

    ranker = HybridRanker(bm25_weight=bm25_weight, vector_weight=vector_weight)

    for query_entry in queries:
        qid = query_entry['qid']
        query = query_entry['query']
        print(f"\nProcessing Query ID: {qid}, Query: {query}")

        query_input = [query] if isinstance(query, str) else query
        results_hybrid = se.search(query_input, use_bm25=use_bm25, use_bert=use_bert, top_n=top_n)

        if use_bm25 and use_bert:
            results_bm25 = se.search(query_input, use_bm25=True, use_bert=False, top_n=top_n)
            results_bert = se.search(query_input, use_bm25=False, use_bert=True, top_n=top_n)
            ranked_results = ranker.rank_documents(results_bm25, results_bert)
        else:
            ranked_results = results_hybrid

        save_trec_run(ranked_results, qid, "hybrid", run_file)

        print(f"\n=== Results for Query ID: {qid} ===")
        for i, doc in enumerate(ranked_results[:5], 1):
            score = doc.get('combined_score', doc.get('score', 0.0))
            doc_id = doc.get('id', 'UNKNOWN')
            title = doc.get('title', 'No Title')
            print(f"Rank {i}: {title} (ID: {doc_id}, Score: {score:.4f})")

    if os.path.exists(run_file) and os.path.exists(qrels_file):
        print("\n=== Evaluating Results ===")
        aggregated_metrics, per_query_results = evaluate_trec(qrels_file, run_file, verbose=verbose)
    else:
        print("Run file or qrels file missing. Skipping evaluation.")
        aggregated_metrics, per_query_results = {}, {}

    return aggregated_metrics, per_query_results

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    deepseek_model, deepseek_tokenizer = deepseek_model(device).get_model()
    summarizer = BartSummarizer("cpu")
    
    model = scibert_model(device)
    print("Initializing preprocess system...")
    preprocess = preprocess_sys(model, deepseek_model, deepseek_tokenizer, device)
    print("Initializing search engine...")
    se = search_engine.engine(model)

    parser = argparse.ArgumentParser(description="Run the search engine with parameters.")
    parser.add_argument("--query_file", type=str, default="test_set_queries.json", help="Path to query JSON file")
    parser.add_argument("--qrels_file", type=str, default="test_set_qrels.txt", help="Path to qrels file")
    parser.add_argument("--use_bm25", action="store_true", help="Enable BM25-based search", default=True)
    parser.add_argument("--use_bert", action="store_true", help="Enable BERT-based semantic search", default=False)
    parser.add_argument("--use_expansion", action="store_true", help="Query expansion")
    parser.add_argument("--exp_syn", action="store_true", help="Apply synonyms expansion")
    parser.add_argument("--exp_sem", action="store_true", help="Query semantic expansion")
    parser.add_argument("--top_n", type=int, default=5, help="Max number of documents to return")
    parser.add_argument("--use_summary", action="store_true", help="Enable BART summarization")
    parser.add_argument("--bm25_weight", type=float, default=0.7, help="Weight for BM25 in hybrid ranking")
    parser.add_argument("--vector_weight", type=float, default=0.3, help="Weight for vector search in hybrid ranking")
    parser.add_argument("--sem_method", type=int, default=2, help="0: Database-vector, 1: GoogleNews-vectors, 2: GenAI")
    parser.add_argument("--verbose", action="store_true", help="Show per-query metrics")
    
    args = parser.parse_args()

    queries = load_queries(args.query_file, max_queries=1000)  # Load only first 100 queries
    print(f"Loaded {len(queries)} queries")

    if args.use_expansion:
        if not args.exp_syn and not args.exp_sem:
            print("Please specify expansion method by --exp_syn & --exp_sem")
            exit()
        processed_queries = []
        for query_entry in queries:
            processed_query = preprocess.process_query(
                query_entry['query'], 
                use_semantic=args.exp_sem, 
                use_synonyms=args.exp_syn, 
                sem_method=args.sem_method
            )
            processed_queries.append({'qid': query_entry['qid'], 'query': processed_query})
        aggregated_metrics, per_query_results = process_queries(
            se, 
            processed_queries, 
            args.use_bm25, 
            args.use_bert, 
            args.top_n, 
            summarizer if args.use_summary else None, 
            args.bm25_weight, 
            args.vector_weight, 
            args.qrels_file, 
            verbose=args.verbose
        )
    else:
        aggregated_metrics, per_query_results = process_queries(
            se, 
            queries, 
            args.use_bm25, 
            args.use_bert, 
            args.top_n, 
            summarizer if args.use_summary else None, 
            args.bm25_weight, 
            args.vector_weight, 
            args.qrels_file, 
            verbose=args.verbose
        )