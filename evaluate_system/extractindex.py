# import json
# from multiprocessing import Pool, cpu_count

# def process_line(line):
#     try:
#         paper = json.loads(line.strip())
#         if paper['categories'] == 'cs.AI':
#             return {
#                 'id': paper['id'],
#                 'title': paper['title'],
#                 'abstract': paper['abstract']
#             }
#         else:
#             return None
#     except json.JSONDecodeError:
#         return None

# def extract_arxiv_fields(input_file, output_file, search_engine=None, num_workers=None):
#     """
#     Extract id, title, and abstract from arXiv JSON data using multiprocessing.
#     """
#     if num_workers is None:
#         num_workers = cpu_count()

#     with open(input_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     print(f"Processing {len(lines)} lines with {num_workers} workers...")

#     # Use multiprocessing to process lines
#     with Pool(num_workers) as pool:
#         results = pool.map(process_line, lines)

#     # Filter out None (failed parses)
#     extracted_data = [paper for paper in results if paper]

#     # Optional indexing
#     if search_engine:
#         for paper in extracted_data:
#             search_engine.index_document(paper)

#     # Write to output file
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(extracted_data, f, indent=2)

#     print(f"Extracted {len(extracted_data)} papers to {output_file}")

# # Example usage standalone
# if __name__ == "__main__":
#     input_file = './database/data/arxiv-metadata-oai-snapshot.json'
#     output_file = './eval_arxiv_data.json'
#     extract_arxiv_fields(input_file, output_file)



from elasticsearch import Elasticsearch, helpers
import json

def extract_from_elasticsearch(index_name="arxiv_index", output_file="extracted_data_eval.json", es_host="http://localhost:9200"):
    """
    Extracts id, title, and abstract from all documents in an Elasticsearch index.
    
    Args:
        index_name (str): The name of the index to query.
        output_file (str): The JSON file to write the results.
        es_host (str): Elasticsearch host URL.
    """
    es = Elasticsearch(es_host)

    # Query to match all documents
    query = {
        "query": {
            "match_all": {}
        }
    }

    print(f"Extracting documents from index: {index_name}...")

    # Use helpers.scan to efficiently scroll through the index
    results = []
    for doc in helpers.scan(es, index=index_name, query=query, scroll='2m'):
        source = doc['_source']
        result = {
            "id": doc['_id'],
            "title": source.get("title", ""),
            "abstract": source.get("abstract", "")
        }
        results.append(result)

    # Write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(results)} documents to {output_file}")

# Example usage
if __name__ == "__main__":
    extract_from_elasticsearch()