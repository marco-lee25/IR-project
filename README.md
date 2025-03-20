
# IR Project - Search Engine 

## Environment setup
Before running the project, ensure you have **Conda** (for Python environment management) and **Docker** (to run the Elasticsearch server) installed.

### Setting Up the Python Environment
```bash
# Create a new Conda environment
conda create --name ir-project python=3.10

# Activate the environment
conda activate ir-project
```

### Installing Required Libraries
```bash
# Navigate to the project directory
cd IR-project

# Install dependencies
pip install -r requirements.txt
```

## The elasticsearch server docker
This project uses **Elasticsearch** for indexing and retrieving documents. The preprocessed indices are stored in `./database/data/`:

**Preprocessed Elasticsearch Indices**:
- `arxiv_index_data.json`
- `arxiv_index_mapping.json`
- `arxiv_index_settings.json`
  
### Preprocessed indices information
When building the docker, the indices will be restore using the script ` /database/import_index.py `. There are currently in total 1000 documents, with topics named `cs.AI` from https://www.kaggle.com/datasets/Cornell-University/arxiv

### Setup docker for elasticsearch server
```bash
  docker build -t ir-project .
  docker run -d --name elasticsearch -p 9200:9200 ir-project
  
  # Optional to check if any error
  docker logs -f elasticsearch
```
### Rebuild the indexing system
If you want to rebuild the indexing system with different name, number of documents, or specify index method, run the python ` /database/process_data.py `, you can edit `use_bert` and `max_doc` inside the file:
```python
if __name__=="__main__":
    index_name = "arxiv_index"
    use_bert = True
    max_doc=1000
    build_index_system(index_name, use_bert, max_doc)
```






