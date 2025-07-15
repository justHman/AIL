import pandas as pd
from . import models 

class Recommender:
    def __init__(self, df):
        self.df = df
        self.user_ids = df['user'].unique()
        self.ratings = df['rating'].unique()
        self.item_ids = df['item'].unique()

        # Các thành phần được tính toán trước
        self.svd_predicted_ratings = None # Sẽ được tính khi init
        self.content_similarity_matrix = None # Sẽ được tính khi init
        self.cf_item_similarity_matrix = None # Sẽ được tính khi init
        print("Recommender is ready.")
    
    def recommend(self, user_id, n=10, aglorithm='hybrid'):
        if user_id not in self.user_ids:
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
            recs = models.iiCB(user_id, n)

        elif aglorithm == 'uuCB':
            pass

        return {'recommendations': recs, 'strategy': 'hybrid_svd_item_cf'}

if __name__ == '__main__':
    df = pd.read_csv('data\captone_data.csv')
    recommender_engine = Recommender(df)

    user_1 = 1889878 # Một user đã có trong hệ thống
    recs_for_user_1 = recommender_engine.recommend(user_1, n=5, aglorithm='knnCF')
    print(f"Gợi ý cho người dùng cũ {user_1}:")
    print(recs_for_user_1)
   