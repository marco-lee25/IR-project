import json
import re

def fuzzy_match(query, text):
    query = query.lower().strip()
    words = query.split()
    return any(re.search(rf"\b{re.escape(word[:5])}", text.lower()) for word in words)

def evaluate_fuzzy(filepath, top_k=5):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    total_precision = 0

    for item in data:
        query = item["query"]
        keyword = query.lower().strip()
        results = item["results"][:top_k]
        matches = 0

        print(f"\nQuery: {query}")
        for i, res in enumerate(results):
            abstract = res.get("abstract", "")
            if fuzzy_match(keyword, abstract):
                matches += 1
                relevance = "Match"
            else:
                relevance = "Not Relevant"

            print(f"\nResult {i+1}: {res['title']}")
            print(f"Relevance: {relevance}")

        precision = matches / top_k
        total_precision += precision
        total += 1
        print(f"\n Precision@{top_k}: {precision:.2f}")

    avg = total_precision / total if total else 0
    print(f"\n Average Precision@{top_k}: {avg:.4f}")

if __name__ == "__main__":
    evaluate_fuzzy("evaluate_system/regex_eval_input.json", top_k=5)
