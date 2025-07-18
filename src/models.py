import pandas as pd
import numpy as np
from src.utils import get_items_rated_by_user
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.linear_model import Ridge
import math
from src.utils import load_config
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score 

config = load_config()
if not config:
    print('Chưa có config!')
    exit(0)

PATH_SIM_MATRIX = config['path_sim_matrix']
PATH_DATA = config['path_data']



# ===================================================================
# 1. MÔ HÌNH PHI CÁ NHÂN HÓA (DÙNG CHO NGƯỜI DÙNG MỚI)
# ===================================================================

def PB(df, top_n=10, return_df=False):
    item_stats = df.groupby('item').agg(
        count=('rating', 'count'),
        avg_rating=('rating', 'mean')
    )
    
    popular_items = item_stats.sort_values(by=['avg_rating', 'count'], ascending=[False, False]).head(top_n)
    
    if return_df:
        return popular_items.index.tolist(), popular_items
    
    return popular_items.index.tolist()

# ===================================================================
# 2. MÔ HÌNH LỌC DỰA TRÊN NỘI DUNG (CONTENT-BASED / ITEM-ITEM) 
# ===================================================================
class Ridge_iiCB:
    def __init__(self, df):
        self.df = df

        self.items = df['item'].unique()
        
        self.users = df['user'].unique()
        self.map_users = {user: i for i, user in enumerate(self.users)}
        
        self.X = None
        self.W = None
        self.b = None            
        self.y = None

    def train(self, vectors):
        self.X = vectors
        self.W = np.zeros((vectors.shape[1], len(self.users))) # 83 x 33901
        self.b = np.zeros((1, len(self.users)))                # 1  x 33901

        for user, id in self.map_users.items():
            items, ratings = get_items_rated_by_user(self.df, user)   
            X_train = self.X.loc[items, :]

            rg = Ridge(alpha=0.01, fit_intercept=True)
            rg.fit(X_train, ratings)
            
            self.W[:, id] = rg.coef_
            self.b[0, id] = rg.intercept_
            
        self.y = self.X @ self.W + self.b
        print('='*50, 'Train xong!', '='*50)

    def predict(self, item, user):
        if self.y is None:
            return 
        return self.y.loc[item, self.map_users[user]]

    def reccomend(self, user, n, return_rating=False):
        if self.y is None:
            return
        
        rated_items = set(self.df[self.df['user'] == user]['item'])
        all_items = set(self.items)
        unrated_items = all_items - rated_items

        item_scores = [(item, float(self.predict(item, user)), len(self.df[self.df['item'] == item])) for item in unrated_items]

        item_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)

        if return_rating:
            return item_scores[:n]
        return [item for item, rating, quantity in item_scores[:n]]
    
    def evaluate(self, sparse, return_result=False):
        if self.y is None:
            return
        
        truth = []
        predict = []
        for user in self.users:
            items, ratings = get_items_rated_by_user(sparse, user)
            ratings_pred = self.predict(items, user)
            truth.extend(ratings)
            predict.extend(ratings_pred)

        mae = mean_absolute_error(truth, predict)
        mse = mean_squared_error(truth, predict)
        r2 = r2_score(truth, predict)
        rmse = np.sqrt(mse)

        if return_result:
            return pd.DataFrame({
                'truth': truth,
                'predict': predict
            }), mae, mse, rmse, r2

        return mae, mse, rmse, r2
         
class iiCB:
    def __init__(self, df, sim_matrix=None):
        self.df = df
        self.sim_matrix = sim_matrix
        
        users = df['user'].unique()
        self.user_map = {user: i for i, user in enumerate(users)}

    def recommend(self, user, n, return_result=False):
        if self.sim_matrix is not None:
            item_total_counts = self.df.groupby('item')['rating'].count().reset_index()
            item_total_counts.columns = ['item', 'total_count']

            user_ratings = self.df[self.df['user'] == user][['item', 'rating']]

            item_stats = pd.merge(user_ratings, item_total_counts, on='item')
            item_stats_sorted = item_stats.sort_values(by=['rating', 'total_count'], ascending=False)

            love_item = item_stats_sorted.iloc[0]['item']
            # love_item = self.df[self.df['user'] == user][['rating', 'item']].sort_values(by='rating', ascending=False)['item'].iloc[0]
            dic = self.sim_matrix[love_item].sort_values(ascending=False)[1:n+1].to_dict()
            
            if return_result:
                return dic
            return list(dic.keys()) 

# ===================================================================
# 3. MÔ HÌNH LỌC CỘNG TÁC (COLLABORATIVE FILTERING)
# ===================================================================
class knnCF:
    def __init__(self, sparse, iiCF_simi_matrix, utility_norm):
        self.sparse = sparse
        self.mean_ratings = self.sparse.groupby('item')['rating'].mean()

        self.users = sorted(sparse['user'].unique())
        self.user_map = {user: i for i, user in enumerate(self.users)}

        self.items = sparse['item'].unique()

        self.iiCF_simi_matrix = iiCF_simi_matrix
        self.utility_norm = utility_norm


    def predict(self, item, user, k=9):
        if item not in self.items or user not in self.users:
            return 0
        
        items_rated_user = self.sparse[self.sparse['user'] == user]['item'].unique()
        if len(items_rated_user) == 0:
            return 0 
        
        sim_scores = self.iiCF_simi_matrix.loc[item, items_rated_user]
        nearest_s = sim_scores.sort_values(ascending=False).head(k)

        # Lấy rating của người dùng cho item
        ratings = []
        for i in nearest_s.index:
            r = self.utility_norm.loc[i, user]
            ratings.append(r)
        ratings = np.array(ratings)

        pred_rating = np.dot(ratings, nearest_s.values) / (np.abs(nearest_s.values).sum() + 1e-8)
        return pred_rating
    
    def evaluate(self, sparse, k=9, return_result=False):
        truth = []
        predict = []
        for user in self.users:
            items, ratings = get_items_rated_by_user(sparse, user)
            if len(items) == 0:
                continue

            ratings_pred = [self.predict(item, user, k) + self.mean_ratings[item] for item in items]
            truth.extend(ratings)
            predict.extend(ratings_pred)

        mae = mean_absolute_error(truth, predict)
        mse = mean_squared_error(truth, predict)
        r2 = r2_score(truth, predict)
        rmse = np.sqrt(mse)

        if return_result:
            return pd.DataFrame({
                'truth': truth,
                'predict': predict
            }), mae, mse, rmse, r2
        return mae, mse, rmse, r2

    def recommend(self, user, n, return_result=False):
        rated_items = set(self.sparse[self.sparse['user'] == user]['item'])
        all_items = set(self.items)
        unrated_items = all_items - rated_items
        rec_items = []
        for item in unrated_items:
            rating_pred = self.predict(item, user)
            if rating_pred >= 0:
                rec_items.append((item, rating_pred))
        rec_items.sort(key=lambda x: x[1], reverse=True)

        if return_result:
            return rec_items[:n]
        return  [item for item, rating in rec_items[:n]]

# def knnCF(user_id, n, ratings_df, k_neighbors=8):
#     train_matrix = ratings_df.pivot_table(index='user', columns='item', values='rating').fillna(0)

#     item_user_matrix = train_matrix.T
    
#     user_ratings = train_matrix.loc[user_id]
#     rated_items = user_ratings[user_ratings > 0].index.tolist()
#     unrated_items = train_matrix.columns.difference(rated_items)

#     knn_model = knn(metric='cosine', algorithm='brute', n_neighbors=k_neighbors + 1)
#     knn_model.fit(item_user_matrix.values)

#     predictions = {}

#     for item in unrated_items:
#         if item not in item_user_matrix.index:
#             continue

#         item_idx = item_user_matrix.index.get_loc(item)
#         distances, indices = knn_model.kneighbors([item_user_matrix.iloc[item_idx]], n_neighbors=k_neighbors + 1)

#         neighbors = item_user_matrix.index[indices[0][1:]]
#         neighbor_ratings = user_ratings[neighbors]
#         neighbor_ratings = neighbor_ratings[neighbor_ratings > 0]

#         if len(neighbor_ratings) > 0:
#             predicted_rating = neighbor_ratings.mean()
#         else:
#             predicted_rating = user_ratings[user_ratings > 0].mean() if user_ratings[user_ratings > 0].any() else 2.5

#         predictions[item] = predicted_rating

#     sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
#     recommended_items = [item for item, _ in sorted_preds[:n]]
#     return recommended_items

def dcpCF(user_id, n):
    """
    Hàm này làm gì:
        Sử dụng phương pháp phân rã ma trận - là 1 trong 2 phương pháp của Colabtive Filltering.
        Sử dụng kết quả từ một mô hình Phân rã Ma trận (như SVD) để đưa ra gợi ý.
        Đây là phương pháp Lọc cộng tác hiện đại và hiệu quả.

    Input:
        - user_id (int/str): ID của người dùng cần gợi ý.
        - n (int): Số lượng item cần gợi ý.
        - Có thể modify thêm biến đề hoàn thành hàm này (vẫn giữ nguyên user_id và n)

    Output:
        - list: Danh sách các item ID được gợi ý.
    """
    # Logic ví dụ: Sắp xếp các rating dự đoán cho user, loại bỏ item đã xem
    return ['item_id_SVD1', 'item_id_SVD2', ...]

if __name__ == '__main__':
    df = pd.read_csv('data\oversampling_data.csv')
    model = iiCB(df.tail(100), vectors=True)
    
    model.recommend(user_id=2, n=10)
