import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


def create_iiCF_simi_matrix(sparse):
    sparse_norm = sparse.copy()
    mean_ratings = sparse_norm.groupby('item')['rating'].mean()
    sparse_norm['rating'] = sparse_norm.apply(
                lambda row: row['rating'] - mean_ratings[row['item']],
                axis=1
    )
    utility_norm = sparse_norm.pivot(index='item', columns='user', values='rating').fillna(0)
    iiCF_simi_matrix = cosine_similarity(utility_norm)
    return pd.DataFrame(iiCF_simi_matrix, index=utility_norm.index, columns=utility_norm.index), sparse_norm, utility_norm
    
def create_iiCB_simi_matrix(strings, index, columns):
    vectorizer = TfidfVectorizer()
    vecs = vectorizer.fit_transform(strings) 

    sm_matrix = cosine_similarity(vecs)
    
    return pd.DataFrame(sm_matrix, index=index, columns=columns)

def vectorize_items(processed_items):
    vectorizer = TfidfVectorizer()
    vecs = vectorizer.fit_transform(processed_items) 

    return pd.DataFrame(vecs.toarray(), index=[item.replace(' ', '') for item in processed_items])
                        
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
    path = r'data\captone_data.csv'
    df = pd.read_csv(path)
    series = df['item'].apply(split_item).unique()
    print(series)
    vecs = vectorize_items(series)
    print(vecs.loc['CC0101EN'])