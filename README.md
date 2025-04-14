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
pip install -r requirement.txt
```

## The elasticsearch server docker
This project uses **Elasticsearch** for indexing and retrieving documents. The processed indices are stored in `./database/data/`:

**Elasticsearch Indices strcuture**:
- `arxiv_index_data.json`
- `arxiv_index_mapping.json`
- `arxiv_index_settings.json`


### Setup docker for elasticsearch server
```bash
  docker build -t ir-project_v8 .
  docker run -d --name ir_project -e "discovery.type=single-node" -e "xpack.security.enabled=false" -p 9200:9200 ir-project_v8
  
```
### Build the indexing system
If you want to build the indexing system with, run the python `Rebuild.py `, you can edit `use_bert` and `max_doc` inside the file:
```python
if __name__=="__main__":
    index_name = "arxiv_index"
    use_bert = True
    max_doc=2000
    build_index_system(index_name, use_bert, max_doc)
```
## Search Example
```bash
cd IR-project
python main.py "face identify" --use_bm25 --use_bert --top_n 5 --use_expansion --exp_sem 
```
Output :
```bash
Initalizing search engine...
Performing semantic query expansion using word2vec
Expanded terms before limit: ['face indentify', 'face locate', 'face pinpoint', 'face uncover', 'face toidentify', 'face indentified', 'face define', 'face detect', 'face classify']
Expanded terms: ['face identify', 'face indentify', 'face locate', 'face pinpoint']
Query expansion result : ['face identify', 'face indentify', 'face locate', 'face pinpoint']
Query: ['face identify', 'face indentify', 'face locate', 'face pinpoint']
BM25: True, Vector: True
Hybrid search
Elasticsearch server is running.
Index 'arxiv_index' exists with 6001 documents.
==================================================
RESULT 1:
Title: Comparing Robustness of Pairwise and Multiclass Neural-Network Systems
  for Face Recognition
Abstract:   Noise, corruptions and variations in face images can seriously hurt the
performance of face recognition systems. To make such systems robust,
multiclass neuralnetwork classifiers capable of learning...
BM25: 17.040 (norm: 1.000)
Vector: 2.563 (norm: 1.000)
Combined: 1.000
--------------------------------------------------
RESULT 2:
Title: Discovering Markov Blanket from Multiple interventional Datasets
Abstract:   In this paper, we study the problem of discovering the Markov blanket (MB) of
a target variable from multiple interventional datasets. Datasets attained from
interventional experiments contain riche...
BM25: 16.969 (norm: 0.993)
Vector: 2.561 (norm: 0.997)
Combined: 0.995
--------------------------------------------------
RESULT 3:
Title: Philosophy in the Face of Artificial Intelligence
Abstract:   In this article, I discuss how the AI community views concerns about the
emergence of superintelligent AI and related philosophical issues.
...
BM25: 14.214 (norm: 0.738)
Vector: 2.534 (norm: 0.964)
Combined: 0.851
--------------------------------------------------
RESULT 4:
Title: Quadratic Unconstrained Binary Optimization Problem Preprocessing:
  Theory and Empirical Analysis
Abstract:   The Quadratic Unconstrained Binary Optimization problem (QUBO) has become a
unifying model for representing a wide range of combinatorial optimization
problems, and for linking a variety of discipli...
BM25: 14.032 (norm: 0.721)
Vector: 2.531 (norm: 0.959)
Combined: 0.840
--------------------------------------------------
RESULT 5:
Title: About Tau-Chain
Abstract:   Tau-chain is a decentralized peer-to-peer network having three unified faces:
Rules, Proofs, and Computer Programs, allowing a generalization of virtually
any centralized or decentralized P2P networ...
BM25: 13.594 (norm: 0.681)
Vector: 2.529 (norm: 0.956)
Combined: 0.818
```


