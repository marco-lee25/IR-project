import pandas as pd
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

# Load JSON
with open('extracted_data_eval.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Clean text
def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum() and t not in stopwords.words('english')]
    return ' '.join(tokens)

df['clean_title'] = df['title'].apply(clean_text)
df['clean_abstract'] = df['abstract'].apply(clean_text)

# Extract queries
kw_model = KeyBERT()
def extract_query(abstract):
    keywords = kw_model.extract_keywords(abstract, keyphrase_ngram_range=(1, 2), top_n=1)
    return keywords[0][0] if keywords else ""

df['query'] = df['abstract'].apply(extract_query)

# Vectorize documents
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(df['clean_abstract'])

# Compute relevant documents with similarity scores
def get_relevant_docs(query, vectorizer, doc_vectors, threshold=0.2):
    query_vec = vectorizer.transform([clean_text(query)])
    similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    # Return list of (index, score) pairs above threshold, sorted by score descending
    return sorted([(i, score) for i, score in enumerate(similarities) if score > threshold], 
                  key=lambda x: x[1], reverse=True)

df['relevant_docs'] = df['query'].apply(lambda q: get_relevant_docs(q, vectorizer, doc_vectors))

# Save documents
df[['id', 'title', 'abstract']].to_json('test_set_documents.json', orient='records')

# Save queries
queries = pd.DataFrame({'qid': range(1, len(df) + 1), 'query': df['query']})
queries.to_json('test_set_queries.json', orient='records')

# Save qrels with up to 3 relevant documents per query
qrels = [{'qid': 'qid', 'iter': 'Q0', 'docid': 'docid', 'rel': 'rel'}]  # Dummy header
for i, row in df.iterrows():
    qid = str(i + 1)
    source_doc_id = row['id']
    # Start with source document
    relevant_doc_ids = [source_doc_id]
    # Add up to 2 more from relevant_docs, excluding source and avoiding duplicates
    for doc_idx, score in row['relevant_docs']:
        doc_id = df.iloc[doc_idx]['id']
        if doc_id != source_doc_id and doc_id not in relevant_doc_ids:
            relevant_doc_ids.append(doc_id)
            if len(relevant_doc_ids) == 5:  # Stop at 3
                break
    # Add each relevant document to qrels
    for doc_id in relevant_doc_ids:
        qrels.append({'qid': qid, 'iter': 'Q0', 'docid': doc_id, 'rel': '1'})
pd.DataFrame(qrels).to_csv('test_set_qrels.txt', sep=' ', index=False, header=False)