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
        # self.word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

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
    
    def expand_semantic_word2vec(self, query, top_k=None):
        if not top_k:
            top_k = self.semantic_topk
        expanded_terms = [query]
        words = query.split()

        for word in words:
            if word in self.word2vec:
                similar_words = self.word2vec.most_similar(word, topn=top_k)
                # Combine with other query words for phrases
                other_words = [w for w in words if w != word]
                for sim_word, _ in similar_words:
                    if other_words:
                        expanded_terms.append(f"{sim_word} {other_words[0]}")
                    else:
                        expanded_terms.append(sim_word)

        print("Expanded terms:", expanded_terms)
        return list(dict.fromkeys(expanded_terms))[:top_k + 1]

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
    
    # #　Version 3
    # def expand_semantic(self, query, top_k=None, max_sentence=5, max_sentence_per_doc=2):
    #     if not top_k:
    #         top_k = self.semantic_topk
    #     expanded_terms = [query]
    #     query_embedding = self.model.encode(query)

    #     similarities = np.dot(self.corpus_embeddings, query_embedding) / (
    #         np.linalg.norm(self.corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
    #     )
    #     all_paragraphs = [(sim, self.corpus_texts[i]) for i, sim in enumerate(similarities)]
    #     all_paragraphs.sort(key=lambda x: x[0], reverse=True)

    #     selected_paragraphs = []
    #     doc_counts = {}
    #     for sim, text in all_paragraphs[:max_sentence * max_sentence_per_doc]:  # Limit candidates
    #         doc_id = self.corpus_doc_ids[all_paragraphs.index((sim, text))]
    #         if len(selected_paragraphs) >= max_sentence:
    #             break
    #         doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
    #         if doc_counts[doc_id] <= max_sentence_per_doc:
    #             selected_paragraphs.append(text)

    #     # Extract candidate phrases from paragraphs and rank by similarity to query
    #     candidate_phrases = []
    #     for paragraph in selected_paragraphs:
    #         phrases = paragraph.split()
    #         for i in range(len(phrases)):
    #             for j in range(i + 1, min(i + 4, len(phrases) + 1)):  # 1-3 word phrases
    #                 phrase = " ".join(phrases[i:j])
    #                 candidate_phrases.append(phrase)

    #     # Rank candidates by similarity to query
    #     phrase_embeddings = self.model.encode(candidate_phrases)
    #     query_emb_norm = np.linalg.norm(query_embedding)
    #     similarities = [np.dot(emb, query_embedding) / (np.linalg.norm(emb) * query_emb_norm) 
    #                     for emb in phrase_embeddings]
    #     top_indices = np.argsort(similarities)[::-1][:top_k]
    #     expanded_terms.extend([candidate_phrases[i] for i in top_indices])

    #     return list(dict.fromkeys(expanded_terms))

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

    # Version 1
    # def expand_semantic(self, query, top_k=None):
    #     if not top_k:
    #         top_k = self.semantic_topk 

    #     expanded_terms = [query]
    #     query_embedding = self.model.encode(query)

    #     # Compute cosine similarity with precomputed embeddings
    #     similarities = np.dot(self.corpus_embeddings, query_embedding) / (
    #         np.linalg.norm(self.corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
    #     )

    #     # Group paragraphs by document and select top paragraphs per document
    #     doc_paragraphs = {}
    #     for idx, (doc_id, sim) in enumerate(zip(self.corpus_doc_ids, similarities)):
    #         if doc_id not in doc_paragraphs:
    #             doc_paragraphs[doc_id] = []
    #         doc_paragraphs[doc_id].append((sim, self.corpus_texts[idx]))

    #     # Select top-k documents based on their highest paragraph similarity
    #     doc_scores = {doc_id: max([sim for sim, _ in paragraphs]) for doc_id, paragraphs in doc_paragraphs.items()}
    #     top_doc_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)[:self.max_sentence_per_doc]

    #     # Extract keyphrases from all paragraphs in top-k documents
    #     all_keyphrases = []
    #     extractor = YAKE()
    #     for doc_id in top_doc_ids:
    #         paragraphs = doc_paragraphs[doc_id]  # List of (similarity, text) tuples
    #         for sim, paragraph in paragraphs:
    #             extractor.load_document(paragraph)
    #             extractor.candidate_selection()
    #             extractor.candidate_weighting()
    #             # Extract self.pke_max_phrases keyphrases per paragraph with their scores
    #             keyphrases = extractor.get_n_best(n=self.pke_max_phrases)
    #             # Store keyphrase with its YAKE score and original paragraph similarity
    #             for kp, score in keyphrases:
    #                 all_keyphrases.append((kp, score, sim))

    #     # Sort all keyphrases by a combined score (e.g., YAKE score + similarity) and select top_k
    #     # You can adjust the scoring logic here (e.g., use only YAKE score or similarity)
    #     sorted_keyphrases = sorted(all_keyphrases, key=lambda x: x[1] + x[2], reverse=True)[:top_k]
    #     top_keyphrases = [kp for kp, _, _ in sorted_keyphrases]

    #     expanded_terms.extend(top_keyphrases)
    #     return list(dict.fromkeys(expanded_terms))  # Remove duplicates
    

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
