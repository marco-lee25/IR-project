import json
import pandas as pd
from utils import *
from sklearn.feature_extraction.text import TfidfVectorizer

# topics = ['cs.AI', 'cs.CV', 'cs.IR', 'cs.LG', 'cs.CL']
topics = ['cs.AI']
current_dir = os.path.dirname(os.path.abspath(__file__))

def load_data():
    cols = ['id', 'title', 'abstract', 'categories']
    data = []
    file_name = os.path.join(current_dir, 'data','arxiv-metadata-oai-snapshot.json')

    with open(file_name, encoding='latin-1') as f:
        for line in f:
            doc = json.loads(line)
            if doc['categories'] in topics:
                lst = [doc['id'], doc['title'], doc['abstract'], doc['categories']]
                data.append(lst)

    df_data = pd.DataFrame(data=data, columns=cols)
    print("Number of data loaded : ", df_data.shape[0])
    # filtered_data = df_data[df_data['categories'].isin(topics)]
    return df_data

def tfidf(prepared_text):
    # Initialize a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer to the 'prepared_text' column
    vectorizer.fit(prepared_text)

    # Transform the documents into TF-IDF vectors
    tfidf_vectors = vectorizer.transform(df_data['prepared_text'])
    
    return tfidf_vectors



if __name__=="__main__":
    download_data()
    df_data = load_data()
    df_data['title'] = df_data['title'].apply(clean_text)
    df_data['abstract'] = df_data['abstract'].apply(clean_text)
    df_data['prepared_text'] = df_data['title'] + ' \n ' + df_data['abstract']
    tfidf_vectors = tfidf(df_data['prepared_text'])