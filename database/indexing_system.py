from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


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

# Indexes documents into Elasticsearch with citations & embeddings.
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
            "citations": {"type": "keyword"},
            }
        }
    }
    if use_bert:
        index_body["mappings"]["properties"]["embedding"] = {"type": "dense_vector", "dims": 768}  # SciBERT has 768 dimensions

        
    response = es.indices.create(index=index_name, body=index_body, ignore=400)

    # Index documents
    if use_bert:
        actions = [
            {
                "_index": index_name,
                "_id": row["id"],
                "_source": {
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "prepared_text": row["prepared_text"],
                    "citations": row["citations"],
                    "embedding": row["embedding"].tolist()  # Convert numpy array to list
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
                    "citations": row["citations"],
                }
            }
            for _, row in df.iterrows()
        ]

    bulk(es, actions)
    print(f"Indexed {len(df)} documents into Elasticsearch.")

def delete_elasticsearch_index(index_name="arxiv_index"):
    # Deletes an index from Elasticsearch.
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"Index '{index_name}' deleted successfully.")
    else:
        print(f"Index '{index_name}' does not exist.")