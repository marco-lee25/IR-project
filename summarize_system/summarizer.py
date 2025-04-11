# summarizer.py

import torch
from transformers import BartTokenizer, BartForConditionalGeneration

class BartSummarizer:
    def __init__(self, device):
        self.device = device
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(self.device)

    def summarize(self, text, max_length=130, min_length=30):
        torch.cuda.empty_cache()
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
