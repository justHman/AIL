import pandas as pd
from . import models 
import os
from src.utils import load_config
from src.data_processing import split_item, create_iiCB_simi_matrix, vectorize_items, create_iiCF_simi_matrix

config = load_config()
PATH_SIM_MATRIX = config['path_sim_matrix']

class Recommender:
    def __init__(self, sparse):
        self.sparse = sparse
        self.users = sparse['user'].unique()
        self.ratings = sparse['rating'].unique()
        self.items = sparse['item'].unique()

        # Các thành phần được tính toán trước
        self.svd_predicted_ratings = None # Sẽ được tính khi init
        self.cf_item_similarity_matrix = None # Sẽ được tính khi init

        items = sparse['item'].unique()
        processed_items = [split_item(item) for item in items]
        if os.path.exists(PATH_SIM_MATRIX):
            iiCB_sim_matrix = pd.read_csv(PATH_SIM_MATRIX)
        else:
            iiCB_sim_matrix = create_iiCB_simi_matrix(processed_items, index=items, columns=items)
        
        self.iiCB_sim_matrix = iiCB_sim_matrix
        self.item_vectors = vectorize_items(processed_items)

        self.iiCF_sim_matrix, self.sparse_norm, self.utility_norm = create_iiCF_simi_matrix(sparse)

        print("Recommender is ready.")
    
    def recommend(self, user, n=10, aglorithm='hybrid'):
        if user not in self.users:
            print(f"\nUser {user} is a new user. Applying cold-start strategy.")
            recs = models.PB(self.sparse, top_n=n)
            return {'recommendations': recs, 'strategy': 'cold_start_popular'}

        # === KỊCH BẢN 2: NGƯỜI DÙNG CŨ ===
        print(f"\nUser {user} is an existing user. Applying {aglorithm} strategy.")
        if aglorithm == 'dcpCF':
            recs = models.dcpCF(user, n)

        elif aglorithm == 'knnCF':
            model = models.knnCF(self.sparse, self.iiCF_sim_matrix, self.utility_norm)
            recs = model.recommend(user, n)
            
        elif aglorithm == 'hybrid':
            item_cf_recs = models.knnCF(user, self.sparse, n)
            svd_recs = models.dcpCF(user, n)
            recs = list(dict.fromkeys(svd_recs + item_cf_recs))[:n]

        elif aglorithm == 'iiCB':
            model = models.iiCB(self.sparse, sim_matrix=self.iiCB_sim_matrix)
            recs = model.recommend(user, n, return_result=False)

        elif aglorithm == 'ridge_iiCB':
            model = models.Ridge_iiCB(self.sparse)
            model.train(self.item_vectors)
            recs = model.reccomend(user, n)

        return {'recommendations': recs, 'strategy': aglorithm}

if __name__ == '__main__':
    df = pd.read_csv('data\captone_data.csv')
    recommender_engine = Recommender(df)
   