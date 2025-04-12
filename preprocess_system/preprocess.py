from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.corpus import wordnet
from pke.unsupervised import YAKE
import torch
import numpy as np
import json
import os
import pickle
from sentence_transformers import models, SentenceTransformer
from transformers import logging
from keybert import KeyBERT
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string
import re

logging.set_verbosity_error()

class preprocess_sys():
    def __init__(self, model, deepseek_model, deepseek_tokenizer, device, json_file="./database/data/arxiv_index_data.json", output_file="./database/data/sentence_corpus.pkl"):
        self.device = device
        self.index_data_path = json_file
        self.corpus_path = output_file
        self.candidate_terms = None
        self.candidate_embeddings = None

        self.pke_max_phrases = 1
        self.max_sentence_per_doc = 3
        self.max_sentence = 2000
        self.semantic_topk = 5
        self.synoyms_topk = 3
        self.yake_window = 2
        self.yake_max_ngram = 3

        self.extractor = YAKE()
        self.model = model

        # # Load DeepSeek-R1-Distill-Qwen-1.5B model and tokenizer
        # self.deepseek_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        # self.deepseek_tokenizer = AutoTokenizer.from_pretrained(self.deepseek_model_name)
        # self.deepseek_model = AutoModelForCausalLM.from_pretrained(self.deepseek_model_name).to(self.device)
        # self.deepseek_model.eval()

        self.deepseek_model = deepseek_model
        self.deepseek_tokenizer = deepseek_tokenizer

        # # Load Gemma-2-2B model and tokenizer
        # self.gemma_model_name = "google/gemma-2-2b"
        # self.gemma_tokenizer = AutoTokenizer.from_pretrained( self.gemma_model_name)
        # self.gemma_model = AutoModelForCausalLM.from_pretrained( self.gemma_model_name).to(self.device)
        # self.gemma_model.eval()

        # # Load TinyLlama-1.1B model and tokenizer
        # self.tinyllama_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        # self.tinyllama_tokenizer = AutoTokenizer.from_pretrained(self.tinyllama_model_name)
        # self.tinyllama_model = AutoModelForCausalLM.from_pretrained(self.tinyllama_model_name).to(self.device)
        # self.tinyllama_model.eval()

        # Load GoogleNews-vectors-negative300 with limited vocabulary
        print("Loading GoogleNews-vectors-negative300 embeddings...")
        self.word2vec = KeyedVectors.load_word2vec_format('./preprocess_system/GoogleNews-vectors-negative300.bin', binary=True, limit=1000000)
        self.vocab_words = self.word2vec.index_to_key  # List of words
        # self.word2vec_vectors = torch.tensor(self.word2vec.vectors, device=self.device)  # Convert to GPU tensor

        self.word2vec_min_sim = 0.5
        self.word2vec_topn = 12

        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

        self.load_corpus()

    def _set_batch_size(self, vector_size=300):
        """Dynamically set batch size based on available GPU memory."""
        if self.device != "cuda":
            return 10000

        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        print(f"Total GPU memory: {total_memory / 1024**3:.2f} GB")

        allocated_memory = torch.cuda.memory_allocated(self.device)
        print(f"Allocated memory after loading models: {allocated_memory / 1024**3:.2f} GB")

        reserved_memory = 1 * 1024**3  # Reserve 1 GB for other operations
        available_memory = total_memory - allocated_memory - reserved_memory
        print(f"Available memory for Word2Vec: {available_memory / 1024**3:.2f} GB")

        vector_size_bytes = vector_size * 4  # 300 dimensions, 4 bytes per float
        max_batch_size = int(available_memory / (vector_size_bytes + 4))
        max_batch_size = max(1000, min(max_batch_size, 10000))  # Clamp between 1000 and 10000

        print(f"Setting batch size to: {max_batch_size}")
        return max_batch_size

    def expand_multiple_queries(self, query_list, use_synonyms=False):
        results = {}
        for q in query_list:
            results[q] = self.process_query(q, use_semantic=True, use_synonyms=use_synonyms)
        return results

    def load_corpus(self):
        with open(self.corpus_path, "rb") as f:
            semantic_data = pickle.load(f)
            self.corpus_texts = semantic_data["texts"]
            self.corpus_embeddings = semantic_data["embeddings"]
            self.corpus_doc_ids = semantic_data["doc_ids"]

    def expand_semantic_deepseek(self, query, top_k=None): 
        """Expand query using DeepSeek-R1-Distill-Qwen-1.5B generative model."""
        if not top_k:
            top_k = self.semantic_topk  # e.g., 10
        # # Stricter prompt to enforce list format without any additional text
        # prompt = (
        #     f"Generate {top_k} related keywords for the given query \"{query}\" as a list on computer science area. Please do not included any reasoning on the output. Just return a list of terms"
        # )        

        prompt = (
            f"Generate exactly {top_k} related keywords in English for the query: '{query}' with similar length in the field of computer science. "
            "Return only a Python-style list of terms, with no additional text, reasoning, chain-of-thought, explanations, or numbering. "
            "Example format: ['term1','term2','term3',...,'termN'], the list should be in one sentence without skip line"
        )
        
        # prompt = (
        #     "=== INSTRUCTIONS ===\n"
        #     f"1. Generate exactly {top_k} keywords related to the query.\n"
        #     "2. Keywords must be in computer science and similar in length to the query.\n"
        #     "3. Return ONLY a Python list like ['term1', 'term2'].\n"
        #     "4. DO NOT write ANY other text, reasioning, explanations, or sentences.\n\n"
        #     "=== EXAMPLES ===\n"
        #     "Input: 'neural network'\n"
        #     "Output: ['CNN', 'RNN', 'transformer', 'MLP', 'LSTM']\n\n"
        #     "Input: 'face identify'\n"
        #     "Output: ['facial recognition', 'face detection', 'biometric ID', 'face matching']\n\n"
        #     "=== YOUR TASK ===\n"
        #     f"Input: '{query}'\n"
        #     "Output:"
        # )

        inputs = self.deepseek_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

        # # Tokens to penalize (e.g., "\n", " explanation", " first,")
        # bad_words = ["\n", " explanation", " reasoning", " okay,", " so,", " think"]
        # bad_word_ids = [self.deepseek_tokenizer.encode(word, add_special_tokens=False) for word in bad_words]
        # bad_word_ids = [item for sublist in bad_word_ids for item in sublist]  # Flatten


        # Generate related queries with settings to enforce concise output
        outputs = self.deepseek_model.generate(
            inputs["input_ids"],
            max_length=500,
            num_return_sequences=1,  # Generate one sequence with the list
            num_beams=1,  # Reduced beams for more focused output
            no_repeat_ngram_size=2,  # Increased to avoid repetition
            early_stopping=True,
            temperature=0.2,  # Lower temperature for more deterministic output
            top_p=0.85,  # Tighter nucleus sampling for focused results
            # do_sample=False,  # Disable sampling for more predictable results
        )

        # Decode and clean up generated queries
        generated_text = self.deepseek_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Debug: Print the raw generated text before filtering
        print("Raw generated text:", generated_text)
        # exit()

        # Remove the prompt part if it appears in the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        list_str = None
        for line in generated_text.split("\n"):
            list_start = line.find('[')
            list_end = line.find(']') + 1
            if list_start != -1 and list_end != 0:
                list_str = line[list_start:list_end]

        if list_str is None:
            print("Cannot generate query...")
            expanded_terms = [query]
        else:
            # Strip brackets and split
            terms_list = list_str.strip("[]").split(", ")
            # Clean each term and remove duplicates
            terms_list = [
                term.strip(' "\'').lower()  # Remove quotes and spaces
                for term in terms_list 
                if term.strip() and term.strip(' "\'').lower() != query.lower()
            ]
            # Combine while removing duplicates
            expanded_terms = list(dict.fromkeys([query] + terms_list[:top_k]))

        return expanded_terms
    
    def expand_semantic_gemma(self, query, top_k=None):
        if not top_k:
            top_k = self.semantic_topk  # e.g., 10
        expanded_terms = [query]

        # Prompt for Gemma to generate related queries
        prompt = f"Generate {top_k} related search queries for '{query}' in the format:\n- query1\n- query2\n- query3\nOutput only the list, nothing else."
        inputs = self.gemma_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

        # Generate related queries
        outputs = self.gemma_model.generate(
            inputs["input_ids"],
            max_length=150,
            num_return_sequences=1,
            num_beams=3,
            no_repeat_ngram_size=3,
            early_stopping=True,
            temperature=0.4,
            top_p=0.85
        )

        # Decode and clean up generated queries
        generated_text = self.gemma_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        lines = [line.strip() for line in generated_text.split("\n") if line.strip()]
        generated_queries = []

        for line in lines:
            line = re.sub(r'^- |^\d+\.\s*', '', line).strip()
            if (line and line != query and line not in generated_queries and
                len(line.split()) <= 5 and
                len(line.split()) >= 1 and
                not re.search(r'[.!?]', line) and
                re.match(r'^[a-zA-Z0-9\s-]+$', line)):
                generated_queries.append(line)

        expanded_terms.extend(generated_queries[:top_k])
        print("Gemma-2-2B expanded terms:", expanded_terms)
        return list(dict.fromkeys(expanded_terms))

    def expand_semantic_tinyllama(self, query, top_k=None):
        """Expand query using TinyLlama-1.1B generative model."""
        if not top_k:
            top_k = self.semantic_topk  # e.g., 10
        expanded_terms = [query]

        # # Prompt with examples to guide TinyLlama
        # prompt = (
        #     f"Generate exactly {top_k} related keywords for the query: '{query}' with similar length in the field of computer science. "
        #     "Return ONLY a Python-style list of terms "
        #     "DO NOT include any additional text, explanations, reasoning, chain-of-thought, or numbering. "
        #     "Example format: ['term1', 'term2', 'term3']"
        # )
        prompt = (
            f"Generate exactly {top_k} related keywords for the query: '{query}' "
            "in the field of computer science. "
            "Return ONLY a Python-style list of terms.\n"
            "Example: ['face recognition', 'object detection', 'image classification']\n"
        )
        
        inputs = self.tinyllama_tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate related queries with sampling for diversity
        outputs = self.tinyllama_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=self.tinyllama_tokenizer.eos_token_id
        )

        # Decode and clean up generated queries
        generated_text = self.tinyllama_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        # if generated_text.startswith(prompt):
        #     generated_text = generated_text[len(prompt):].strip()

        # Debug: Print the raw generated text before filtering
        print("Raw generated text:", generated_text)
        exit()

        lines = [line.strip() for line in generated_text.split("\n") if line.strip()]
        generated_queries = []

        # Encode the original query for relevance check
        query_embedding = self.model.encode(query.lower())

        for line in lines:
            line = re.sub(r'^- |^\d+\.\s*', '', line).strip()
            # Relaxed filtering to allow more outputs for debugging
            if (line and line != query and line not in generated_queries and
                len(line.split()) <= 5 and
                len(line.split()) >= 1):
                # Relevance check
                line_embedding = self.model.encode(line.lower())
                similarity = np.dot(query_embedding, line_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(line_embedding))
                if similarity > 0.4:  # Lowered threshold for debugging
                    generated_queries.append(line)

        # Debug: Print the filtered queries
        print("Filtered queries:", generated_queries)

        expanded_terms.extend(generated_queries[:top_k])
        print("TinyLlama-1.1B expanded terms:", expanded_terms)
        exit()
        return list(dict.fromkeys(expanded_terms))

    def expand_semantic_word2vec(self, query, top_k=3, topn=None, min_similarity=None):
        print("Performing semantic query expansion using GoogleNews-vectors-negative300 on GPU")
        if topn is None:
            topn = self.word2vec_topn
        if min_similarity is None:
            min_similarity = self.word2vec_min_sim

        words = query.lower().split()
        preprocessed_ori_words = [self.preprocess_text(word) for word in words]
        expanded_terms = [query]
        related_words = {}

        # Load vectors onto GPU dynamically
        if self.device == "cuda":
            print("Moving GoogleNews vectors to GPU...")
            word2vec_vectors = torch.tensor(self.word2vec.vectors, device=self.device)
        else:
            word2vec_vectors = None

        if self.device == "cuda" and word2vec_vectors is not None:
            print("Using GPU for expansion with batch processing")
            batch_size = self._set_batch_size(vector_size=300)  # 300 dimensions for GoogleNews

            for id, word in enumerate(words):
                if word in self.vocab_words:
                    word_idx = self.vocab_words.index(word)
                    word_vec = word2vec_vectors[word_idx].unsqueeze(0)  # Already on GPU
                    related_words[word] = []

                    # Process vectors in batches
                    for i in range(0, len(self.vocab_words), batch_size):
                        batch_vectors = word2vec_vectors[i:i + batch_size]  # Already on GPU
                        similarities = torch.cosine_similarity(word_vec, batch_vectors)
                        scores, indices = torch.topk(similarities, k=min(topn + 1, batch_vectors.shape[0]), largest=True)

                        for score, idx in zip(scores[1:], indices[1:]):  # Skip self
                            w = self.vocab_words[i + idx.item()]
                            if score.item() >= min_similarity and self.preprocess_text(w) != preprocessed_ori_words[id] and w != word:
                                related_words[word].append(w)

                    torch.cuda.empty_cache()  # Clear temporary GPU memory
        else:
            print("Using CPU for expansion with GoogleNews embeddings")
            for id, word in enumerate(words):
                if word in self.word2vec:
                    related_words[word] = []
                    similar = self.word2vec.most_similar(word, topn=topn)
                    for w, score in similar:
                        if score >= min_similarity and self.preprocess_text(w) != preprocessed_ori_words[id]:
                            related_words[word].append(w)

        # Generate expansions with balanced representation
        if len(words) > 1:
            # Expand each word independently
            for i, orig_word in enumerate(words):
                if orig_word in related_words:
                    for new_word in related_words[orig_word]:
                        new_term = words.copy()
                        new_term[i] = new_word
                        expanded_terms.append(" ".join(new_term))
            
            # Add combinations of both words changing
            if len(words) == 2 and words[0] in related_words and words[1] in related_words:
                for w1 in related_words[words[0]]:
                    for w2 in related_words[words[1]]:
                        expanded_terms.append(f"{w1} {w2}")
        else:
            expanded_terms.extend(related_words.get(words[0], []))

        # Filter and limit
        expanded_terms = [term for term in expanded_terms if term != query and term]
        print("Expanded terms before limit:", expanded_terms)

        # Ensure original query is first, then take top_k additional terms
        final_terms = [query] + [term for term in expanded_terms if term != query][:top_k]
        print("Expanded terms:", final_terms)

        # Free GPU memory
        if self.device == "cuda":
            del word2vec_vectors
            torch.cuda.empty_cache()

        return list(dict.fromkeys(final_terms))  # Remove duplicates, preserve order
    
    def expand_semantic(self, query, top_k=None, max_sentence=5, max_sentence_per_doc=2):
        if not top_k:
            top_k = self.semantic_topk 

        expanded_terms = [query]
        query_embedding = self.model.encode(query.lower())

        similarities = np.dot(self.corpus_embeddings, query_embedding) / (
            np.linalg.norm(self.corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        doc_paragraphs = {}
        for idx, (doc_id, sim) in enumerate(zip(self.corpus_doc_ids, similarities)):
            if doc_id not in doc_paragraphs:
                doc_paragraphs[doc_id] = []
            doc_paragraphs[doc_id].append((sim, self.corpus_texts[idx]))

        selected_paragraphs = []
        doc_counts = {}

        all_paragraphs = []
        for doc_id, paragraphs in doc_paragraphs.items():
            for sim, text in paragraphs:
                all_paragraphs.append((sim, text, doc_id))
        all_paragraphs.sort(key=lambda x: x[0], reverse=True)

        for sim, text, doc_id in all_paragraphs:
            if len(selected_paragraphs) >= max_sentence:
                break
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            if doc_counts[doc_id] <= max_sentence_per_doc:
                selected_paragraphs.append((sim, text))

        all_keyphrases = []
        for sim, paragraph in selected_paragraphs:
            self.extractor.load_document(paragraph)
            self.extractor.candidate_selection(n=self.yake_max_ngram)
            self.extractor.candidate_weighting(window=self.yake_window)
            keyphrases = self.extractor.get_n_best(n=self.pke_max_phrases)
            for kp, score in keyphrases:
                all_keyphrases.append((kp, score, sim))

        sorted_keyphrases = sorted(all_keyphrases, key=lambda x: x[1] + x[2], reverse=True)[:top_k]
        top_keyphrases = [kp for kp, _, _ in sorted_keyphrases]

        print("Top keyphrases selected:", top_keyphrases)
        expanded_terms.extend(top_keyphrases)
        return list(dict.fromkeys(expanded_terms))

    def preprocess_text(self, text, mode='title'):
        text = text.lower()
        return self.stemmer.stem(text) 
        tokens = [self.stemmer.stem(word) for word in text.split()]
        if mode == "title":
            text = text.lower()
            tokens = [self.lemmatizer.lemmatize(word) for word in text.split()]
            tokens = " ".join(tokens)
            return tokens
        else:
            result = []
            tmp = text.split('\n')
            text = [i.strip() for i in tmp if i.strip() != '']
            text = ''.join(text)
            text = re.sub(r'\.([A-Z])', r'. \1', text)
            sentences = sent_tokenize(text)
            for sentence in sentences:
                tokens = word_tokenize(sentence) 
                tokens = [lemmatizer.lemmatize(word.lower().translate(str.maketrans("", "", string.punctuation))) for word in tokens if word not in stop_words]
                tokens = " ".join(tokens)
                result.append(tokens)
            return result

    def expand_synoyms(self, query, top_k=None):
        if not top_k:
            top_k = self.synoyms_topk

        ori_words = query.split()
        expand_words = ori_words.copy()
        for word in ori_words:
            syn_list = set()
            for sys in wordnet.synsets(word):
                for lemma in sys.lemmas():
                    tmp_word = lemma.name()
                    if not tmp_word in syn_list:
                        syn_list.add(tmp_word)

            syn_list = sorted(list(syn_list - {word}))[:top_k]
            expand_words.extend(syn_list)
        return expand_words

    def process_query(self, query, use_semantic=True, use_synonyms=False, sem_method=0):
        expanded_query = None 
        if use_semantic:
            if sem_method == 0:
                expanded_query = self.expand_semantic(query)
            elif sem_method == 1:
                expanded_query = self.expand_semantic_word2vec(query)
            elif sem_method == 2:
                expanded_query = self.expand_semantic_deepseek(query)

                
        if use_synonyms:
            if expanded_query:
                for i in self.expand_synoyms(query):
                    expanded_query.append(i)
            else:
                expanded_query = self.expand_synoyms(query)

        return expanded_query