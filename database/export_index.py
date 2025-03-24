# import json
# import requests
# import os 

# # Elasticsearch URL
# ES_HOST = "http://localhost:9200"
# INDEX_NAME = "arxiv_index"

# current_dir = os.path.dirname(os.path.abspath(__file__))
# target_dir = os.path.join(current_dir, "data")

# # Export index mapping
# mapping_response = requests.get(f"{ES_HOST}/{INDEX_NAME}/_mapping")
# mapping = mapping_response.json()
# mapping_save_path = os.path.join(target_dir, f"{INDEX_NAME}_mapping.json")
# with open(mapping_save_path, "w") as f:
#     json.dump(mapping, f, indent=4)

# # Export index settings
# settings_response = requests.get(f"{ES_HOST}/{INDEX_NAME}/_settings")
# settings = settings_response.json()
# settings_save_path = os.path.join(target_dir, f"{INDEX_NAME}_settings.json")
# with open(settings_save_path, "w") as f:
#     json.dump(settings, f, indent=4)

# # Export index data (scroll API)
# scroll_url = f"{ES_HOST}/{INDEX_NAME}/_search?scroll=1m"
# search_body = {"size": 1000, "query": {"match_all": {}}}  # Fetch all documents
# response = requests.get(scroll_url, json=search_body).json()

# documents = []
# while response["hits"]["hits"]:
#     documents.extend(response["hits"]["hits"])
#     scroll_id = response["_scroll_id"]
#     response = requests.post(f"{ES_HOST}/_search/scroll", json={"scroll": "1m", "scroll_id": scroll_id}).json()

# # Save documents
# save_path = os.path.join(target_dir, f"{INDEX_NAME}_data.json")
# with open(save_path, "w") as f:
#     json.dump(documents, f, indent=4)

# print(f"Exported {len(documents)} documents from {INDEX_NAME}.")


import json
import requests
import os

def export_index():
    # Elasticsearch URL
    ES_HOST = "http://localhost:9200"
    INDEX_NAME = "arxiv_index"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "data")
    os.makedirs(target_dir, exist_ok=True)

    # Export index mapping
    mapping_response = requests.get(f"{ES_HOST}/{INDEX_NAME}/_mapping")
    mapping_response.raise_for_status()
    mapping = mapping_response.json()
    mapping_save_path = os.path.join(target_dir, f"{INDEX_NAME}_mapping.json")
    with open(mapping_save_path, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"Exported mapping to {mapping_save_path}")

    # Export index settings (filter out metadata)
    settings_response = requests.get(f"{ES_HOST}/{INDEX_NAME}/_settings")
    settings_response.raise_for_status()
    settings = settings_response.json()
    # Filter out read-only fields
    filtered_settings = {
        INDEX_NAME: {
            "settings": {
                "index": {
                    "number_of_shards": settings[INDEX_NAME]["settings"]["index"]["number_of_shards"],
                    "number_of_replicas": settings[INDEX_NAME]["settings"]["index"]["number_of_replicas"]
                }
            }
        }
    }
    settings_save_path = os.path.join(target_dir, f"{INDEX_NAME}_settings.json")
    with open(settings_save_path, "w") as f:
        json.dump(filtered_settings, f, indent=4)
    print(f"Exported filtered settings to {settings_save_path}")

    # Export index data (scroll API)
    scroll_url = f"{ES_HOST}/{INDEX_NAME}/_search?scroll=1m"
    search_body = {"size": 1000, "query": {"match_all": {}}}
    response = requests.get(scroll_url, json=search_body)
    response.raise_for_status()
    response = response.json()

    documents = []
    while response["hits"]["hits"]:
        documents.extend(response["hits"]["hits"])
        scroll_id = response["_scroll_id"]
        response = requests.get(f"{ES_HOST}/_search/scroll", json={"scroll": "1m", "scroll_id": scroll_id})
        response.raise_for_status()
        response = response.json()

    # Save documents
    save_path = os.path.join(target_dir, f"{INDEX_NAME}_data.json")
    with open(save_path, "w") as f:
        json.dump(documents, f, indent=4)
    print(f"Exported {len(documents)} documents to {save_path}")
