import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def create_similarity_matrix(strings, index, columns, return_vector=False):
    vectorizer = TfidfVectorizer()
    vecs = vectorizer.fit_transform(strings) 

    sm_matrix = cosine_similarity(vecs)
    
    if return_vector:
        return pd.DataFrame(sm_matrix, index=index, columns=columns), vecs.toarray()
    
    return pd.DataFrame(sm_matrix, index=index, columns=columns)


def split_item(item):
    # Các trường hợp đặc biệt cần xử lý thủ công
    exceptions = {
        'RAVSCTEST1': 'RAVSC TEST 1',
        'BENTEST4': 'BEN TEST 4',
        'GPXX0QR3EN': 'GPXX 0 QR3 EN',
        'GPXX0T0FEN': 'GPXX 0 T0F EN'
    }
    if item in exceptions:
        return exceptions[item]

    # Tách EN và vX nếu có
    item = re.sub(r'(EN)(v\d+)?$', r' \1 \2', item)

    # Tách prefix chữ đầu khỏi phần còn lại
    item = re.sub(r'^([A-Z]+)([^A-Z].*)', r'\1 \2', item)

    # 👉 Bổ sung: Tách cụm số 4 chữ số thành 2 cụm 2 chữ số
    item = re.sub(r'\b(\d{2})(\d{2})\b', r'\1 \2', item)

    # Tách phần còn lại thành các khối logic
    tokens = re.findall(r'[A-Z]+|\d+|[A-Z]+\d+|\d+[A-Z]+|EN|v\d+', item)

    return ' '.join(filter(None, tokens))

if __name__ == '__main__':
    from utils import create_similarity_matrix
    path = r'data\captone_data.csv'
    df = pd.read_csv(path)
    series = df['item'].unique()
    strings = preprocess_items(df)
    sm_matrix = create_similarity_matrix(strings, index=series, columns=series)
    print(sm_matrix)