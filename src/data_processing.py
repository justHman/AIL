import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_item_similarity_matrix(path_data, path_folder='data\similarity_matrix'):
    df = pd.read_csv(path_data)

    items = df['item'].unique() # ['DS0101EN',...]
    strings = []
    for item in items:
        split_item = [item[i:i+2] for i in range(0, len(item), 2)]
        string = ' '.join(split_item) # DS 01 01 EN
        strings.append(string)        # ['DS 01 01 EN',...]

    vectorizer = TfidfVectorizer()
    vecs = vectorizer.fit_transform(strings) 

    sm_matrix = cosine_similarity(vecs)

    path_save = path_folder + 'item-item.csv'
    pd.DataFrame(sm_matrix, index=items, columns=items).to_csv(path_save)
    print(f'Lưu thành công similarity_matrix vào: {path_save}')
    return path_save

if __name__ == '__main__':
    create_item_similarity_matrix('data\captone_data.csv')