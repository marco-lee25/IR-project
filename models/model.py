from sentence_transformers import models, SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

class scibert_model():
    def __init__(self, device):
        self.device = device
        word_embedding_model = models.Transformer(
            'allenai/scibert_scivocab_uncased',
            max_seq_length=128,
            do_lower_case=True
        )
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(self.device)

    
    def encode(self, query):
        return self.model.encode(query)
    

class deepseek_model():
    def __init__(self, device):
        self.deepseek_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.deepseek_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.deepseek_model_name).to(device)
        self.model.eval()

    def get_model(self):
        return self.model, self.tokenizer