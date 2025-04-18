import json
import requests

ES_HOST = "http://localhost:9200"
INDEX_NAME = "arxiv_index"

# Load settings and mapping
with open(f"/usr/share/elasticsearch/{INDEX_NAME}_settings.json", "r") as f:
    settings = json.load(f)
with open(f"/usr/share/elasticsearch/{INDEX_NAME}_mapping.json", "r") as f:
    mapping = json.load(f)

# Combine settings and mapping into one payload
index_body = {
    "settings": settings[INDEX_NAME]["settings"],
    "mappings": mapping[INDEX_NAME]["mappings"]
}


# Check if the index already exists
check = requests.head(f"{ES_HOST}/{INDEX_NAME}")
if check.status_code == 200:
    print(f"Index '{INDEX_NAME}' already exists. Skipping creation and import.")
    exit(0)

print("Index body being sent:", json.dumps(index_body, indent=4))

# Create index with settings and mapping together
response = requests.put(f"{ES_HOST}/{INDEX_NAME}", json=index_body)
if response.status_code != 200:
    print(f"Failed to create index: {response.status_code}, {response.text}")
    exit(1)
print(f"Created index: {response.status_code}, {response.text}")

# Load and import documents
with open(f"/usr/share/elasticsearch/{INDEX_NAME}_data.json", "r") as f:
    documents = json.load(f)

batch_size = 100
bulked = 0

for i in range(0, len(documents), batch_size):
    bulk_data = ""
    for doc in documents[i:i + batch_size]:
        bulk_data += json.dumps({"index": {"_index": INDEX_NAME, "_id": doc["_id"]}}) + "\n"
        bulk_data += json.dumps(doc["_source"]) + "\n"

    bulk_response = requests.post(
        f"{ES_HOST}/_bulk",
        data=bulk_data,
        headers={"Content-Type": "application/x-ndjson"}
    )

    if bulk_response.status_code not in (200, 201):
        print(f"Failed to bulk insert batch {i // batch_size + 1}: {bulk_response.status_code}, {bulk_response.text}")
        exit(1)

    print(f"Batch {i}, Index restored: {bulk_response.status_code}, {bulk_response.text}")