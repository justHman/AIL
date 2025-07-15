# from src.data_processing import preprocess_items
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def create_similarity_matrix(strings, index, columns):
    vectorizer = TfidfVectorizer()
    vecs = vectorizer.fit_transform(strings) 

    sm_matrix = cosine_similarity(vecs)

    return pd.DataFrame(sm_matrix, index=index, columns=columns)

# def load_sm_matrix(path_load, path_create):
#     if path_load:
#         return pd.read_csv(path_load)
#     if 
