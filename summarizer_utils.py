# summarizer.py

import torch
from transformers import BartTokenizer, BartForConditionalGeneration

class BartSummarizer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(self.device)

    def summarize(self, text, max_length=130, min_length=30):
        if not text.strip():
            return "No content to summarize."
        inputs = self.tokenizer([text], max_length=1024, return_tensors='pt', truncation=True).to(self.device)
        summary_ids = self.model.generate(
            inputs['input_ids'],
            num_beams=4,
            min_length=min_length,
            max_length=max_length,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)



from summarizer import Summarizer

# Load model once (global)
bert_model = Summarizer()

def summarize_text(text, query=None, ratio=0.3):
    """
    Summarizes text using BERT summarizer.
    If a query is provided, makes the summary more query-aware.
    """
    # if query:
    #     summary = bert_model(text, query=query, ratio=ratio)
    # else:
    summary = bert_model(text, ratio=ratio)
    return summary
