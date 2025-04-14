# summarizer.py

import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM

class deepseekSummarizer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    def summarize(self, text):
        print("Summarizing test : ", text)
        print("==============")
        prompt = (
            "Summarize the following text : \n"
            f"{text}"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1000).to(self.device)

        # Generate related queries with settings to enforce concise output
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=100,
            num_return_sequences=1,  # Generate one sequence with the list
            num_beams=1,  # Reduced beams for more focused output
            no_repeat_ngram_size=2,  # Increased to avoid repetition
            early_stopping=True,
            temperature=0.5,  # Lower temperature for more deterministic output
            top_p=0.85,  # Tighter nucleus sampling for focused results
            do_sample=True,  # Disable sampling for more predictable results
        )

        # Decode and clean up generated queries
        print("Decoding.....")
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # Remove the prompt part if it appears in the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        print("Summarized result : ", generated_text)
        exit()
        return generated_text

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
