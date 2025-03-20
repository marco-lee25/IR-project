
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
When building the docker, the indices will be restore using the script ` /database/import_index.py `. There are currently in total 1000 documents, with topics named `cs.AI` from https://www.kaggle.com/datasets/Cornell-University/arxiv

### Setup docker for elasticsearch server
```bash
  docker build -t ir-project .
  docker run -d --name elasticsearch -p 9200:9200 ir-project
  
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


## Search Example
```bash
cd IR-project
python main.py "face identify" --use_bm25 True --use_bert True --top_n 2
```
Output :
```bash
Title: comparing robustness pairwise multiclass neuralnetwork system face recognition
 Abstract: noise corruption variation face image seriously hurt performance face recognition system make system robust multiclass neuralnetwork classifier capable learning noisy data suggested however large face data set system provide robustness high level paper explore pairwise neuralnetwork system alternative approach improving robustness face recognition experiment approach shown outperform multiclass neuralnetwork system term predictive accuracy face image corrupted noise
 bm25_score:8.66293
 vector_score:1.7126617

Title: emotion appraisalcoping model cascade problem
 Abstract: modelling emotion become challenge nowadays therefore several model produced order express human emotional activity however currently able express close relationship existing emotion cognition appraisalcoping model presented aim simulate emotional impact caused evaluation particular situation appraisal along consequent cognitive reaction intended face situation coping model applied cascade problem small arithmetical exercise designed tenyearold pupil goal create model corresponding child behaviour solving problem using strategy
 bm25_score:5.0492687
 vector_score:1.7084448
```



