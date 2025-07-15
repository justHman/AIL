import pandas as pd
from src.utils import create_similarity_matrix

def preprocess_items(df, col='item'):
    series = df[col]
    items = series.unique() # ['DS0101EN',...]
    strings = []
    for item in items:
        split_item = [item[i:i+2] for i in range(0, len(item), 2)]
        string = ' '.join(split_item) # DS 01 01 EN
        strings.append(string)        # ['DS 01 01 EN',...]
    return strings 

if __name__ == '__main__':
    from utils import create_similarity_matrix
    path = r'data\captone_data.csv'
    df = pd.read_csv(path)
    series = df['item'].unique()
    strings = preprocess_items(df)
    sm_matrix = create_similarity_matrix(strings, index=series, columns=series)
    print(sm_matrix)