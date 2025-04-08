import json
from rouge_score import rouge_scorer
from bert_score import score

def evaluate_rouge(summaries_file):
    with open(summaries_file) as f:
        data = json.load(f)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for i, item in enumerate(data):
        scores = scorer.score(item['reference'], item['generated'])
        print(f"\nSample {i+1}")
        for k, v in scores.items():
            print(f"{k}: Precision={v.precision:.2f}, Recall={v.recall:.2f}, F1={v.fmeasure:.2f}")

def evaluate_bertscore(summaries_file):
    with open(summaries_file) as f:
        data = json.load(f)
    refs = [d['reference'] for d in data]
    hyps = [d['generated'] for d in data]
    P, R, F1 = score(hyps, refs, lang="en", verbose=True)
    print(f"\nBERTScore - P: {P.mean():.4f}, R: {R.mean():.4f}, F1: {F1.mean():.4f}")

if __name__ == "__main__":
    evaluate_rouge("evaluation/summaries.json")
    evaluate_bertscore("evaluation/summaries.json")
