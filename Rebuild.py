from database.process_data import build_index_system
from database import export_index
import time
from elasticsearch import Elasticsearch

if __name__ == "__main__":
    build_index_system(index_name="arxiv_index", use_bert=True, max_doc=2000, remove_index=True)