from sentence_transformers import SentenceTransformer
from transformers import *

from nltk.corpus import wordnet
from pke.unsupervised import YAKE


import numpy as np
import json
import os
import pickle
from sentence_transformers import models, SentenceTransformer
from transformers import logging, T5Tokenizer, T5ForConditionalGeneration
from keybert import KeyBERT
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string


logging.set_verbosity_error()
class preprocess_sys():
    def __init__(self, json_file="./database/data/arxiv_index_data.json", output_file="./database/data/sentence_corpus.pkl"):
        self.index_data_path = json_file
        self.corpus_path = output_file
        self.candidate_terms = None
        self.candidate_embeddings = None

        self.pke_max_phrases = 1
        self.max_sentence_per_doc = 3
        self.max_sentence = 2000
        self.semantic_topk = 10

        self.synoyms_topk = 3

        self.yake_window = 2
        self.yake_max_ngram = 3

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


        self.extractor = YAKE()
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        # self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        # self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

        # Download weight from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g
        self.word2vec = KeyedVectors.load_word2vec_format('./preprocess_system/GoogleNews-vectors-negative300.bin', binary=True)
        self.word2vec_min_sim = 0.5
        self.word2vec_topn = 12
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))


        # self.build_sentence_corpus_from_json()
        self.load_corpus()

    def expand_multiple_queries(self, query_list, use_synonyms=False):
        results = {}
        for q in query_list:
            results[q] = self.process_query(q, use_semantic=True, use_synonyms=use_synonyms)
        return results

    def load_corpus(self):
        # Load precomputed term embeddings (from earlier semantic expansion)
        with open(self.corpus_path, "rb") as f:
            semantic_data = pickle.load(f)
            self.corpus_texts = semantic_data["texts"]
            self.corpus_embeddings = semantic_data["embeddings"]
            self.corpus_doc_ids = semantic_data["doc_ids"]
    
    def expand_semantic_word2vec(self, query, top_k=3, topn=None, min_similarity=None):
        print("Performing semantic query expansion using word2vec")
        if topn is None:
            topn = self.word2vec_topn
        if min_similarity is None :
            min_similarity = self.word2vec_min_sim

        words = query.lower().split()

        preprocessed_ori_words = [self.preprocess_text(word) for word in words]
        expanded_terms = [query]
        related_words = {}

        # Get similar words for each term with context
        for id, word in enumerate(words):
            if word in self.word2vec:
                related_words[word]=[]
                similar = self.word2vec.most_similar(word, topn=topn)
                for w, score in similar:
                    tmp_w = w
                    if score >= min_similarity and self.preprocess_text(tmp_w) != preprocessed_ori_words[id] :
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
            if abs(len(words) - len(related_words)) <= 1:
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
        return list(dict.fromkeys(final_terms))  # Remove duplicates, preserve order

    def expand_semantic_T5(self, query, top_k=None):
        if not top_k:
            top_k = self.semantic_topk  # e.g., 3
        expanded_terms = [query]
        
        # Prompt to generate related terms
        prompt = f"{query}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.t5_model.generate(
            inputs["input_ids"],
            max_length=50,
            num_return_sequences=top_k,
            num_beams=top_k + 1,  # Beam search for diversity
            early_stopping=True
        )
        
        # Decode and clean up generated terms
        generated_terms = [self.tokenizer.decode(out, skip_special_tokens=True).strip() 
                          for out in outputs]
        expanded_terms.extend(term for term in generated_terms if term != query and term)
        
        print("Expanded terms:", expanded_terms)
        return list(dict.fromkeys(expanded_terms))  # Remove duplicates

    # Version 2
    def expand_semantic(self, query, top_k=None, max_sentence=5, max_sentence_per_doc=2):
        if not top_k:
            top_k = self.semantic_topk 

        expanded_terms = [query]
        query_embedding = self.model.encode(query.lower())

        # 1. Compute cosine similarity with precomputed embeddings
        similarities = np.dot(self.corpus_embeddings, query_embedding) / (
            np.linalg.norm(self.corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Group paragraphs by document with their similarities
        doc_paragraphs = {}
        for idx, (doc_id, sim) in enumerate(zip(self.corpus_doc_ids, similarities)):
            if doc_id not in doc_paragraphs:
                doc_paragraphs[doc_id] = []
            doc_paragraphs[doc_id].append((sim, self.corpus_texts[idx]))

        # 2. Select up to max_sentence paragraphs, limited to max_sentence_per_doc per document
        selected_paragraphs = []
        doc_counts = {}  # Track number of paragraphs selected per document

        # Sort all paragraphs by similarity across all documents
        all_paragraphs = []
        for doc_id, paragraphs in doc_paragraphs.items():
            for sim, text in paragraphs:
                all_paragraphs.append((sim, text, doc_id))
        all_paragraphs.sort(key=lambda x: x[0], reverse=True)  # Sort by similarity

        # Pick paragraphs while respecting constraints
        for sim, text, doc_id in all_paragraphs:
            if len(selected_paragraphs) >= max_sentence:
                break  # Stop if we’ve reached max_sentence
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            if doc_counts[doc_id] <= max_sentence_per_doc:
                selected_paragraphs.append((sim, text))

        # print("Selected paragraphs:", [text for _, text in selected_paragraphs])

        # 3. Extract pke_max_phrases keyphrases from each selected paragraph and select top_k
        all_keyphrases = []
        for sim, paragraph in selected_paragraphs:
            self.extractor.load_document(paragraph)
            self.extractor.candidate_selection(n=self.yake_max_ngram)
            self.extractor.candidate_weighting(window=self.yake_window)
            # Extract self.pke_max_phrases keyphrases per paragraph
            keyphrases = self.extractor.get_n_best(n=self.pke_max_phrases)
            # Store keyphrase with YAKE score and paragraph similarity
            for kp, score in keyphrases:
                all_keyphrases.append((kp, score, sim))

        # Sort by combined score (YAKE score + similarity) and select top_k
        sorted_keyphrases = sorted(all_keyphrases, key=lambda x: x[1] + x[2], reverse=True)[:top_k]
        top_keyphrases = [kp for kp, _, _ in sorted_keyphrases]

        # all_keyphrases = [(kp, score, sim, self.model.encode(kp)) for kp, score, sim in all_keyphrases]
        # query_emb_norm = np.linalg.norm(query_embedding)
        # sorted_keyphrases = sorted(all_keyphrases, key=lambda x: np.dot(x[3], query_embedding) / (np.linalg.norm(x[3]) * query_emb_norm), reverse=True)[:top_k]
        # top_keyphrases = [kp for kp, _, _, _ in sorted_keyphrases]

        # print("All keyphrases with scores:", all_keyphrases)
        print("Top keyphrases selected:", top_keyphrases)

        expanded_terms.extend(top_keyphrases)
        return list(dict.fromkeys(expanded_terms))  # Remove duplicates

    def preprocess_text(self, text, mode='title'):
        text = text.lower()
        return self.stemmer.stem(text) 
        tokens = [self.stemmer.stem(word) for word in text.split()]

        """Lowercases, tokenizes, removes stopwords, and lemmatizes text."""    
        if mode == "title":
            text = text.lower()  # Lowercasing
            # text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
            # tokens = word_tokenize(text)  # Tokenization
            tokens = [self.lemmatizer.lemmatize(word) for word in text.split()]  # Lemmatization & Stopword Removal
            tokens = " ".join(tokens)
            return tokens
        else:
            result = []
            tmp = text.split('\n')
            # Dropping the unwant elements (space)
            text = [i.strip() for i in tmp if i.strip() != '']
            # Combine to a single string
            text = ''.join(text)
            # Add space after fullstop
            text = re.sub(r'\.([A-Z])', r'. \1', text)
            # Seperate sentences
            sentences = sent_tokenize(text)
            for sentence in sentences:
                tokens = word_tokenize(sentence) 
                tokens = [lemmatizer.lemmatize(word.lower().translate(str.maketrans("", "", string.punctuation))) for word in tokens if word not in stop_words]  # Lemmatization & Stopword Removal
                tokens = " ".join(tokens)
                result.append(tokens)
            return result
    
    def expand_synoyms(self, query, top_k=None):
        if not top_k:
            top_k = self.synoyms_topk

        ori_words = query.split()
        expand_words = ori_words.copy()
        for word in ori_words:
            # Use set for better operation
            syn_list = set()
            for sys in wordnet.synsets(word):
                for lemma in sys.lemmas():
                    tmp_word = lemma.name()
                    if not tmp_word in syn_list:
                        syn_list.add(tmp_word)

            # Get top_k synonyms (not including source word)
            syn_list = sorted(list(syn_list - {word}))[:top_k]
            # syn_list = list(syn_list - {word})[:top_k]
            expand_words.extend(syn_list)
        return expand_words

    def process_query(self, query,  use_semantic=True, use_synonyms=False):   
        expanded_query = None 
        if use_semantic:
            expanded_query = self.expand_semantic_word2vec(query)

        if use_synonyms:
            if expanded_query:
                for i in self.expand_synoyms(query):
                    expanded_query.append(i)
            else :
                expanded_query = self.expand_synoyms(query)

        return expanded_query

# TODO
# Handle mulitple query ?
# test = preprocess_sys()
# queries  = ["face detection", "reinforcement learning", "neural network"]
# expanded = test.expand_multiple_queries(queries)
# for q, terms in expanded.items():
#     print(f"\nExpanded for '{q}':\n{terms}")
