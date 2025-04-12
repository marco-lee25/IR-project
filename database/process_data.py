import os
import json
import pandas as pd
import re
from database.utils import download_data, clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import spacy
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import pickle  # To save TF-IDF indexed data
from database.indexing_system import index_elasticsearch, check_elasticsearch_server, delete_elasticsearch_index, build_sentence_corpus_from_json
from sentence_transformers import models, SentenceTransformer
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import numpy as np 
from database import export_index
import time

from nltk.tokenize import word_tokenize, PunktSentenceTokenizer

topics = ['cs.AI', 'cs.CV', 'cs.IR', 'cs.LG', 'cs.CL']
# topics = ['cs.AI']
current_dir = os.path.dirname(os.path.abspath(__file__))


def load_data(Scibert, max_doc, model=None):
    # Also download necessary NLTK resources
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

    if Scibert :
        cols = ['id', 'title', 'abstract', 'categories', 'prepared_text', 'paragraph_embeddings']
    else:
        cols = ['id', 'title', 'abstract', 'categories', 'prepared_text']

    data = []
    file_name = os.path.join(current_dir, 'data','arxiv-metadata-oai-snapshot.json')

    # Count lines once for tqdm total
    with open(file_name, encoding='latin-1') as f:
        total_lines = sum(1 for _ in f)

    with open(file_name, encoding='latin-1') as f:
        for i, line in enumerate(tqdm(f, total=total_lines, desc="Processing documents")):
            if len(data) > max_doc:
                break
            doc = json.loads(line)

            if doc['categories'] in topics:
                doc_id = str(doc['id'])  # Ensure ID is a string
                processed_title = preprocess_text(doc['title'],mode="title")
                processed_abstract = preprocess_text(doc['abstract'])
                prepared_text = [processed_title] + processed_abstract  # Combined text for better indexing
                if Scibert:
                    # Generates a dense vector representation of text using SciBERT.
                    paragraph_embeddings = [
                        {'text': p, 'embedding': model.encode(p).tolist()}
                        for p in prepared_text if p.strip()
                    ]
                    # sentences = sent_tokenize(prepared_text)
                    # embedding = np.mean(model.encode(sentences), axis=0)

                    data.append([doc_id, doc['title'], doc['abstract'], doc['categories'], prepared_text, paragraph_embeddings])
                else:
                    data.append([doc_id, doc['title'], doc['abstract'], doc['categories'], prepared_text])
    
    df_data = pd.DataFrame(data=data, columns=cols)
    print("Number of documents loaded:", df_data.shape[0])
    return df_data

def tfidf(prepared_text):
    # Initialize a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the 'prepared_text' column
    vectorizer.fit(prepared_text)

    # Transform the documents into TF-IDF vectors
    tfidf_vectors = vectorizer.transform(df_data['prepared_text'])
    
    return tfidf_vectors

def preprocess_text(text, mode=None):
    """Lowercases, tokenizes, removes stopwords, and lemmatizes text."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    if mode == "title":
        text = text.lower()  # Lowercasing
        text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
        # tokens = word_tokenize(text)  # Tokenization

        punkt = PunktSentenceTokenizer()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatization & Stopword Removal
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

def test():
    # Connect to Elasticsearch server
    es = Elasticsearch("http://localhost:9200")

    index_name = "arxiv_index"  # Replace with your index name

    # Initialize a scroll request to get all documents
    scroll_time = "2m"  # Keep scroll context open for 2 minutes
    batch_size = 100  # Retrieve 100 documents per batch

    response = es.search(index=index_name, body={"query": {"match_all": {}}}, scroll=scroll_time, size=batch_size)

    # Extract the scroll ID and the first batch of hits
    scroll_id = response["_scroll_id"]
    documents = response["hits"]["hits"]

    # Retrieve additional batches until all documents are fetched
    while len(response["hits"]["hits"]) > 0:
        response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
        documents.extend(response["hits"]["hits"])

    print(f"Total documents retrieved: {len(documents)}")

    # Example: Print the first few retrieved documents
    for doc in documents[:5]:  # Print only the first 5 for readability
        print(f"ID: {doc['_id']}, Title: {doc['_source']['title']}, Abstract: {doc['_source']['abstract']}")

def build_index_system(index_name = "arxiv_index", use_bert=True, max_doc=500, remove_index=False):
    if use_bert:
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

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    if remove_index:
        delete_elasticsearch_index(index_name)

    if True:
        download_data()
        df_data = load_data(use_bert, max_doc, model)
        # Index into Elasticsearch
        index_elasticsearch(df_data, index_name, use_bert)


    if use_bert:
        time.sleep(10)
        export_index.export_index()
        build_sentence_corpus_from_json(model)

# if __name__=="__main__":
#     index_name = "arxiv_index"
#     use_bert = True
#     max_doc=2000
#     build_index_system(index_name, use_bert, max_doc, remove_index=True)
