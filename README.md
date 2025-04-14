# IR Project - Search Engine - Grp 26

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
- `sentence_corpus.pkl`
  
### Preprocessed indices information
When building the docker, the indices will be restore using the script ` /database/import_index.py `. There are currently in total 300 documents (due to github's file size limitation), with topics named `cs.AI` from https://www.kaggle.com/datasets/Cornell-University/arxiv

### Setup docker for elasticsearch server
```bash
  docker build -t ir-project_v8 .
  docker run -d --name ir_project -e "discovery.type=single-node" -e "xpack.security.enabled=false" -p 9200:9200 ir-project_v8
  
```
### Build the indexing system
We highly recommend you rebuild the indexing system, since there are only 300 indexes in the data currently, it also affects the performance of query expansion(Database-vector-similarity).
**Before you rebuild, please make sure you have placed the 'kaggle.json' at './database/data/`**
```bash
python ./Rebuild.py
```
You can also edit `use_bert` and `max_doc` inside the file:
```python
if __name__=="__main__":
    index_name = "arxiv_index"
    use_bert = True
    max_doc=5000
    build_index_system(index_name, use_bert, max_doc)
```
## Downloading the Word2Vec data
Before running the program, please download the  `GoogleNews-vectors-negative300.bin ` from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g and place it in  `/preprocess_system `

## Query expansion
This project provided 4 different method on query expansion :
1. Synonyms-based: WordNet
2. Semantic-based: Database-vector-similarity
3. Semantic-based: Word2Vec on GoogleNews-vectors-negative300
4. Experiment: DeepSeek query expansion generation.
   
## Search Example with UI
```bash
python UI.py
```
![image](https://github.com/user-attachments/assets/379a91c6-c0d8-4aa7-bd4a-e3e43186f2f3)



## Search Example with cmd
```bash
cd IR-project
python main.py "face identify" --use_bm25 --use_bert --top_n 5 --use_expansion --exp_sem

Parameter include :
    parser.add_argument("--use_bm25", action="store_true", help="Enable BM25-based search")
    parser.add_argument("--use_bert", action="store_true", help="Enable BERT-based semantic search")
    parser.add_argument("--use_expansion", action="store_true", help="Query expansion")
    parser.add_argument("--exp_syn", action="store_true", help="Apply synoyms expansion")
    parser.add_argument("--exp_sem", action="store_true", help="Query semantic expansion")
    parser.add_argument("--top_n", type=int,default=10, help='Max number of documents return')
    parser.add_argument("--use_summary", action="store_true", help='Enable BART summarization')
    parser.add_argument("--bm25_weight", type=float, default=0.7, 
                   help="Weight for BM25 in hybrid ranking (0.0-1.0)")
    parser.add_argument("--vector_weight", type=float, default=0.3,
                   help="Weight for vector search in hybrid ranking (0.0-1.0)")
    parser.add_argument("--sem_method", type=int, default=1,
               help=" 0:Semantic expansion on GoogleNews-vectors\n 1: Expansion using GenAI")
```
Output :
```bash
Initalizing preprocess system...
Loading GoogleNews-vectors-negative300 embeddings...
Initalizing search engine...
Performing semantic query expansion using GoogleNews-vectors-negative300 on GPU
Using CPU for expansion with GoogleNews embeddings
Expanded terms before limit: ['face indentify', 'face locate', 'face pinpoint', 'face uncover', 'face indentified', 'face define', 'face detect', 'face classify', 'face analyze']
Expanded terms: ['face identify', 'face indentify', 'face locate', 'face pinpoint']
Query expansion result : ['face identify', 'face indentify', 'face locate', 'face pinpoint']
Query: ['face identify', 'face indentify', 'face locate', 'face pinpoint']
BM25: True, Vector: True
Hybrid search with weighted query terms
Elasticsearch server is running.
Index 'arxiv_index' exists with 1001 documents.
bm25 only search
Elasticsearch server is running.
Index 'arxiv_index' exists with 1001 documents.
Handling msearch case
Bert only search
Elasticsearch server is running.
Index 'arxiv_index' exists with 1001 documents.
Handling msearch case

=== RANKING COMPARISON ===
BM25 Order                               | Vector Order                             | Hybrid Order                            
------------------------------------------------------------------------------------------------------------------------
Comparing Robustness of Pairwise an...   | Classification of artificial intell...   | Hybrid Tractable Classes of Binary ...   
BM25: 16.61 | Vector: 2.56 | Combined: 0.78
------------------------------------------------------------------------------------------------------------------------
Emotion: Appraisal-coping model for...   | Detection and emergence                  | Multimodal Biometric Systems - Stud...   
BM25: 10.37 | Vector: 2.52 | Combined: 0.77
------------------------------------------------------------------------------------------------------------------------
Emotion : mod\`ele d'appraisal-copi...   | Symmetry within Solutions                | Comparing Robustness of Pairwise an...   
BM25: 10.17 | Vector: 2.52 | Combined: 0.74
------------------------------------------------------------------------------------------------------------------------
Hybrid Tractable Classes of Binary ...   | Multimodal Biometric Systems - Stud...   | When do Numbers Really Matter?           
BM25: 10.01 | Vector: 2.51 | Combined: 0.66
------------------------------------------------------------------------------------------------------------------------
Multimodal Biometric Systems - Stud...   | A Directional Feature with Energy b...   | Back and Forth Between Rules and SE...   
BM25: 9.97 | Vector: 2.50 | Combined: 0.65
------------------------------------------------------------------------------------------------------------------------

=== HYBRID RANKING RESULTS ===
Rank 1: Hybrid Tractable Classes of Binary Quantified Constraint Satisfaction
  Problems
Abstract:   In this paper, we investigate the hybrid tractability of binary Quantified
Constraint Satisfaction Problems (QCSPs). First, a basic tractable class ...
Scores: BM25: 10.01 | Combined: 0.78 | (Norm: BM25=1.00, Vector=0.26)
Summary:  In this paper, we investigate the hybrid tractability of binary Quantified-Constraint Satisfaction Problems (QCSPs) First, a basic tractable class of binary QCSPs is identified by using the broken-triangle property . Second, we break this restriction to allow that thatexistentially quantified variables can be shifted within or out of their blocks . Finally, we identify a more generalized tractable Class: the min-of-max extendable class .
================================================================================
Rank 2: Multimodal Biometric Systems - Study to Improve Accuracy and Performance
Abstract:   Biometrics is the science and technology of measuring and analyzing
biological data of human body, extracting a feature set from the acquired data,
...
Scores: BM25: 4.98 | Vector: 1.79 | Combined: 0.77 | (Norm: BM25=0.76, Vector=0.81)
Summary:  Biometrics is the science and technology of measuring and analyzing the data of human body . Multimodal biometric systems perform better than unimodal systems and are popular even more complex also .
================================================================================
Rank 3: Comparing Robustness of Pairwise and Multiclass Neural-Network Systems
  for Face Recognition
Abstract:   Noise, corruptions and variations in face images can seriously hurt the
performance of face recognition systems. To make such systems robust,
multic...
Scores: BM25: 8.30 | Combined: 0.74 | (Norm: BM25=0.94, Vector=0.26)
Summary:  Noise, corruptions and variations in face images can seriously hurt the performance of face recognition systems . Multiclass neuralnetwork classifiers capable of learning from noisy data have been suggested . However on large face data sets such systems cannot provide the robustness at a high level .
================================================================================
Rank 4: When do Numbers Really Matter?
Abstract:   Common wisdom has it that small distinctions in the probabilities
(parameters) quantifying a belief network do not matter much for the results of
pr...
Scores: BM25: 6.17 | Combined: 0.66 | (Norm: BM25=0.84, Vector=0.26)
Summary:  Small variations in network parameters can lead to significant changes in computations, authors say . Authors: Small differences in probabilities do not matter much for probabilistic queries . They say their analytic results pinpoint some interesting situations under whichparameter changes do or not matter .
================================================================================
Rank 5: Back and Forth Between Rules and SE-Models (Extended Version)
Abstract:   Rules in logic programming encode information about mutual interdependencies
between literals that is not captured by any of the commonly used seman...
Scores: BM25: 5.85 | Combined: 0.65 | (Norm: BM25=0.81, Vector=0.26)
Summary:  Rules in logic programming encode information about mutual interdependencies that is not captured by any of the commonly used semantics . This information becomes essential as soon as a program needs to be modified or further manipulated . We argue that a program should not be viewed solely as the set of settings of its models .
================================================================================

```
