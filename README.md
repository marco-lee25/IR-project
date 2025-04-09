# TODO 
1. Ranking system
2. Refine semantic search and query expansion
3. Summarization on result

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
This project uses **Elasticsearch** for indexing and retrieving documents. The preprocessed indices are stored in `./database/data/`:

**Preprocessed Elasticsearch Indices**:
- `arxiv_index_data.json`
- `arxiv_index_mapping.json`
- `arxiv_index_settings.json`
  
### Preprocessed indices information
When building the docker, the indices will be restore using the script ` /database/import_index.py `. There are currently in total 2000 documents, with topics named `cs.AI` from https://www.kaggle.com/datasets/Cornell-University/arxiv

### Setup docker for elasticsearch server
```bash
  docker build -t ir-project_v8 .
  docker run -d --name ir_project -e "discovery.type=single-node" -e "xpack.security.enabled=false" -p 9200:9200 ir-project_v8
  
```
## Search Example
```bash
cd IR-project
python main.py "face identify" --use_bm25 --use_bert --top_n 2
```
Output :
```bash
Title: comparing robustness pairwise multiclass neuralnetwork system face recognition
 Abstract: noise corruption variation face image seriously hurt performance face recognition system make system robust multiclass neuralnetwork classifier capable learning noisy data suggested however large face data set system provide robustness high level paper explore pairwise neuralnetwork system alternative approach improving robustness face recognition experiment approach shown outperform multiclass neuralnetwork system term predictive accuracy face image corrupted noise     
 bm25_score:8.748668
 vector_score:1.609658

Title: decision flexibility
 Abstract: development new method representation temporal decisionmaking requires principled basis characterizing measuring flexibility decision strategy face uncertainty goal paper provide framework theory observing decision policy behave face informational perturbation gain clue might behave face unanticipated possibly unarticulated uncertainty end find beneficial distinguish two type uncertainty small world large world uncertainty first type resolved posing unambiguous question clairvoyant anchored welldefined aspect decision frame second type troublesome yet often greater interest address issue flexibility type uncertainty resolved consulting psychic next observe one approach flexibility used economics literature already implicitly accounted maximum expected utility meu principle decision theory though simple observation establishes context illuminating notion flexibility term flexibility respect information revelation show perform flexibility analysis static ie single period decision problem using simple example observe flexible alternative thus identified necessarily meu alternative extend analysis dynamic ie multiperiod model demonstrate calculate value flexibility decision strategy allow downstream revision upstream commitment decision
 bm25_score:6.3052607
 vector_score:1.5658739
```

### Rebuild the indexing system
If you want to rebuild the indexing system with different name, number of documents, or specify index method, run the python `Rebuild.py `, you can edit `use_bert` and `max_doc` inside the file:
```python
if __name__=="__main__":
    index_name = "arxiv_index"
    use_bert = True
    max_doc=1000
    build_index_system(index_name, use_bert, max_doc)
```


