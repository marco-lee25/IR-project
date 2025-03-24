from sentence_transformers import SentenceTransformer
from transformers import *

from nltk.corpus import wordnet
from pke.unsupervised import YAKE


import numpy as np
import json
import os
import pickle
from sentence_transformers import models, SentenceTransformer
from transformers import logging
logging.set_verbosity_error()
class preprocess_sys():
    def __init__(self, json_file="./database/data/arxiv_index_data.json", output_file="./database/data/sentence_corpus.pkl"):
        self.index_data_path = json_file
        self.corpus_path = output_file
        self.candidate_terms = None
        self.candidate_embeddings = None

        self.pke_max_phrases = 5
        self.semantic_topk = 3
        self.synoyms_topk = 3
        word_embedding_model = models.Transformer(
            'gsarti/biobert-nli',
            max_seq_length=128,
            do_lower_case=True
        )
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
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
            self.corpus_sentence = semantic_data["sentences"]
            self.corpus_embeddings = semantic_data["embeddings"]

    def expand_semantic(self, query, top_k=None):
        if not top_k:
            top_k = self.semantic_topk 

        expanded_terms = [query]
        query_embedding = self.model.encode(query)

        similarities = np.dot(self.corpus_embeddings, query_embedding) / (
            np.linalg.norm(self.corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        top_indices = np.argsort(similarities)[::-1][:top_k]
        similar_sentences = [self.corpus_sentence[i] for i in top_indices]

        # Since the sentence from abstract are very long comparing to the query.
        # We only extract key phrase from the similar sentences, use pke. 
        extractor = YAKE()
        for sentence in similar_sentences:
            extractor.load_document(sentence)
            extractor.candidate_selection()
            extractor.candidate_weighting()
            keyphrases = [kp for kp, score in extractor.get_n_best(n=self.pke_max_phrases)]
            expanded_terms.extend(keyphrases)

            # expanded_terms.extend(similar_sentences)   
        return list(dict.fromkeys(expanded_terms))

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
            expanded_query = self.expand_semantic(query)

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
