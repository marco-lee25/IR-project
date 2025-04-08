from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
from tqdm import tqdm
import pickle
from nltk.tokenize import sent_tokenize
import nltk
# from export_index import export_index

# Define a global Elasticsearch client
es = Elasticsearch("http://localhost:9200")

def check_indices():
    # Get all indices
    indices = es.indices.get_alias("*")
    print("Existing indices:")
    for index in indices:
        print(index)

def check_elasticsearch_server(index_name="arxiv_index"):
    # Check if Elasticsearch is running
    if es.ping():
        print("Elasticsearch server is running.")
    else:
        print("Elasticsearch server is NOT running. Please run the docker")
        exit()

    # Check if index exists
    if es.indices.exists(index=index_name):
        doc_count = es.count(index=index_name)["count"]
        print(f"Index '{index_name}' exists with {doc_count} documents.")
        return True
    else:
        print(f"Index '{index_name}' does NOT exist.")
        return False

# GPT
# def build_sentence_corpus_from_json(model=None, json_file="./database/data/arxiv_index_data.json", output_file="./database/data/sentence_corpus.pkl"):
#     nltk.download('punkt')

#     # Load exported Elasticsearch data
#     with open(json_file, "r", encoding="utf-8") as f:
#         documents = json.load(f)
    
#     sentences = []
#     print("Extracting sentences from documents...")
#     for doc in tqdm(documents):
#         source = doc.get("_source", {})
#         title = source.get('title', '')
#         abstract = source.get('abstract', '')
#         full_text = f"{title}. {abstract}".strip()

#         # Use NLTK to tokenize into sentences
#         doc_sentences = sent_tokenize(full_text)

#         # Optional: Filter out very short or meaningless sentences
#         doc_sentences = [s.strip() for s in doc_sentences if len(s.strip().split()) >= 5]

#         sentences.extend(doc_sentences)

#     # Remove duplicate sentences
#     sentences = list(dict.fromkeys(sentences))
    
#     print(f"Encoding {len(sentences)} unique sentences...")
#     embeddings = model.encode(sentences, show_progress_bar=True, batch_size=32)
    
#     # Save the encoded corpus
#     with open(output_file, "wb") as f:
#         pickle.dump({"sentences": sentences, "embeddings": embeddings}, f)
    
#     print(f"Saved {len(sentences)} sentences and embeddings to {output_file}")

# GPT new
def build_sentence_corpus_from_json(model=None, json_file="./database/data/arxiv_index_data.json", output_file="./database/data/sentence_corpus.pkl"):
    # Load the JSON data containing documents
    with open(json_file, "r", encoding="utf-8") as f:
        documents = json.load(f)
    
    paragraph_data = []
    print("Extracting paragraph texts, embeddings, and doc_ids from documents...")
    for doc in tqdm(documents):
        source = doc.get("_source", {})
        paragraphs = source.get('paragraphs', [])
        
        # Extract text, embedding, and doc_id from each paragraph
        for paragraph in paragraphs:
            text = paragraph.get('text', '').strip()
            embedding = paragraph.get('embedding', None)
            if text and embedding and len(text.split()) >= 3 :# Ensure both text and embedding exist
                paragraph_data.append({
                    "doc_id": doc["_id"],
                    "text": text,
                    "embedding": embedding
                })

    # Remove duplicates based on text
    unique_paragraphs = {data["text"]: data for data in paragraph_data}.values()
    
    # Separate texts, embeddings, and doc_ids for saving
    texts = [data["text"] for data in unique_paragraphs]
    embeddings = [data["embedding"] for data in unique_paragraphs]
    doc_ids = [data["doc_id"] for data in unique_paragraphs]
    
    # Save to a pickle file with doc_ids included
    with open(output_file, "wb") as f:
        pickle.dump({"texts": texts, "embeddings": embeddings, "doc_ids": doc_ids}, f)
    
    print(f"Saved {len(texts)} unique paragraph texts, embeddings, and doc_ids to {output_file}")

def index_elasticsearch(df, index_name="arxiv_index", use_bert=False):
    if not es.ping():
        print("ERROR: Cannot connect to Elasticsearch. Make sure it's running.")
        return

    index_body = {
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "abstract": {"type": "text"},
            "prepared_text": {"type": "text"},
            }
        }
    }

    if use_bert:
        index_body["mappings"]["properties"]["paragraphs"] = {
                    "type": "nested",
                    "properties": {
                        "text": {"type": "text"},
                        "embedding": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"}
                    }
                }
        
    response = es.indices.create(index=index_name, body=index_body)

    # Index documents
    if use_bert:
        actions = [
            {
                "_index": index_name,
                "_id": row["id"],
                "_source": {
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "prepared_text": " ".join(row["prepared_text"]),  # Convert list to string
                    "paragraphs": row["paragraph_embeddings"]  # Nested field with text and embeddings
                }
            }
            for _, row in df.iterrows()
        ]
    else:
        actions = [
            {
                "_index": index_name,
                "_id": row["id"],
                "_source": {
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "prepared_text": row["prepared_text"],
                }
            }
            for _, row in df.iterrows()
        ]

    # bulk(es, actions)
    bulk(es, actions, chunk_size=500, request_timeout=60)

    print(f"Indexed {len(df)} documents into Elasticsearch.")
    return

def delete_elasticsearch_index(index_name="arxiv_index"):
    # Deletes an index from Elasticsearch.
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Index '{index_name}' deleted successfully.")
    else:
        print(f"Index '{index_name}' does not exist.")



if __name__ == "__main__":
    build_sentence_corpus_from_json()
