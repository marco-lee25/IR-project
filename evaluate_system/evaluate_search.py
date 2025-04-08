import json
from search_engine import run_search

def precision_at_k(retrieved_titles, relevant_titles, k):
    retrieved_k = retrieved_titles[:k]
    relevant_set = set(title.lower() for title in relevant_titles)
    match_count = sum(1 for title in retrieved_k if title.lower() in relevant_set)
    return match_count / k

def evaluate(queries_file, k=5, use_bm25=True, use_bert=True):
    with open(queries_file, 'r') as f:
        queries = json.load(f)

    total_precision = 0
    for q in queries:
        query = q["query"]
        relevant_titles = q["relevant_titles"]

        results = run_search(query, use_bm25=use_bm25, use_bert=use_bert, top_n=k)
        retrieved_titles = [r['title'] for r in results]

        precision = precision_at_k(retrieved_titles, relevant_titles, k)
        total_precision += precision

        print(f"\nQuery: {query}")
        print(f"Precision@{k}: {precision:.2f}")
        print("Top results:")
        for title in retrieved_titles:
            print(f"- {title}")

    avg_precision = total_precision / len(queries)
    print(f"\nAverage Precision@{k}: {avg_precision:.2f}")

if __name__ == "__main__":
    evaluate("evaluation/queries.json", k=5, use_bm25=True, use_bert=True)
