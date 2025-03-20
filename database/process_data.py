import os
import json
import pandas as pd
from utils import download_data, clean_text
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
from indexing_system import index_elasticsearch, check_elasticsearch_server, delete_elasticsearch_index, check_indices
from sentence_transformers import SentenceTransformer

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

max_doc = 20
# topics = ['cs.AI', 'cs.CV', 'cs.IR', 'cs.LG', 'cs.CL']
topics = ['cs.AI']
current_dir = os.path.dirname(os.path.abspath(__file__))

def load_data(Scibert=True):
    if Scibert :
        # Load the SciBERT model
        model = SentenceTransformer("allenai/scibert_scivocab_uncased")
        cols = ['id', 'title', 'abstract', 'categories', 'prepared_text', 'citations', 'embedding']
    else:
        cols = ['id', 'title', 'abstract', 'categories', 'prepared_text', 'citations']

    data = []
    file_name = os.path.join(current_dir, 'data','arxiv-metadata-oai-snapshot.json')

    with open(file_name, encoding='latin-1') as f:
        for i, line in enumerate(f):
            if len(data) > max_doc:
                break
            doc = json.loads(line)
            if doc['categories'] in topics:
                doc_id = str(doc['id'])  # Ensure ID is a string
                processed_title = preprocess_text(doc['title'])
                processed_abstract = preprocess_text(doc['abstract'])
                prepared_text = processed_title + " \n " + processed_abstract  # Combined text for better indexing
                citations = extract_citations(doc['abstract'])  # Extract citations if needed
                if Scibert:
                    #Generates a dense vector representation of text using SciBERT.
                    embedding = model.encode(prepared_text)
                    data.append([doc_id, processed_title, processed_abstract, doc['categories'], prepared_text, citations, embedding])
                else:
                    data.append([doc_id, processed_title, processed_abstract, doc['categories'], prepared_text, citations])

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

def preprocess_text(text):
    """Lowercases, tokenizes, removes stopwords, and lemmatizes text."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()  # Lowercasing
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatization & Stopword Removal
    return " ".join(tokens)

def extract_citations(text):
    # Load Spacy NLP model for citation extraction
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    citations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]  # Example: extract organization references
    return citations

def search_elasticsearch(query, top_n=5):
    """Search for documents in Elasticsearch using BM25 ranking."""
    es = Elasticsearch("http://localhost:9200")  # Connect to Elasticsearch

    search_query = {
        "query": {
            "match": {
                "prepared_text": query  # Search in the indexed text
            }
        },
        "size": top_n  # Return top N results
    }

    response = es.search(index="arxiv_index", body=search_query)
    
    # Extract results
    results = []
    for hit in response["hits"]["hits"]:
        results.append({
            "id": hit["_id"],
            "title": hit["_source"]["title"],
            "abstract": hit["_source"]["abstract"],
            "score": hit["_score"]  # Relevance score
        })
    
    return results

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

if __name__=="__main__":
    use_bert = False
    index_name = "arxiv_index"
    delete_elasticsearch_index(index_name)
    if not check_elasticsearch_server():
        download_data()
        df_data = load_data(use_bert)
        # Index into Elasticsearch
        index_elasticsearch(df_data, index_name, use_bert)

    query = "neuralnetwork"
    results = search_elasticsearch(query)

    for doc in results:
        print(f"Title: {doc['title']}\n Abstract: {doc['abstract']}\n Score: {doc['score']}\n")