import pandas as pd
from . import models 
import os
from src.utils import load_config
from src.data_processing import split_item, create_similarity_matrix, vectorize_items

config = load_config()
PATH_SIM_MATRIX = config['path_sim_matrix']
PATH_VECTORS = config['path_vectors']

class Recommender:
    def __init__(self, df):
        self.df = df
        self.users = df['user'].unique()
        self.ratings = df['rating'].unique()
        self.items = df['item'].unique()

        # Các thành phần được tính toán trước
        self.svd_predicted_ratings = None # Sẽ được tính khi init
        self.cf_item_similarity_matrix = None # Sẽ được tính khi init

        items = df['item'].unique()
        processed_items = [split_item(item) for item in items]
        if os.path.exists(PATH_SIM_MATRIX):
            sim_matrix = pd.read_csv(PATH_SIM_MATRIX)
        else:
            sim_matrix = create_similarity_matrix(processed_items, index=items, columns=items, return_vector=True)
        
        self.sim_matrix = sim_matrix
        self.item_vectors = vectorize_items(processed_items)

        print("Recommender is ready.")
    
    def recommend(self, user_id, n=10, aglorithm='hybrid'):
        if user_id not in self.users:
            print(f"\nUser {user_id} is a new user. Applying cold-start strategy.")
            recs = models.PB(self.df, top_n=n)
            return {'recommendations': recs, 'strategy': 'cold_start_popular'}

        # === KỊCH BẢN 2: NGƯỜI DÙNG CŨ ===
        print(f"\nUser {user_id} is an existing user. Applying {aglorithm} strategy.")
        if aglorithm == 'dcpCF':
            recs = models.dcpCF(user_id, n)

        elif aglorithm == 'knnCF':
            recs = models.knnCF(user_id, n, self.df)
            
        elif aglorithm == 'hybrid':
            item_cf_recs = models.knnCF(user_id, self.df, n)
            svd_recs = models.dcpCF(user_id, n)
            recs = list(dict.fromkeys(svd_recs + item_cf_recs))[:n]

        elif aglorithm == 'iiCB':
            model = models.iiCB(self.df, vectors=self.item_vectors)
            recs = model.reccomend(user_id, n)

        elif aglorithm == 'ridge_iiCB':
            model = models.Ridge_iiCB(self.df)
            model.train(self.item_vectors)

        return {'recommendations': recs, 'strategy': aglorithm}

if __name__ == '__main__':
    df = pd.read_csv('data\captone_data.csv')
    recommender_engine = Recommender(df)
   