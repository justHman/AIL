import pandas as pd
import numpy as np
from src.utils import get_items_rated_by_user
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.linear_model import Ridge
import math
from src.utils import load_config


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
class iiCB:
    def __init__(self,  df, sim_matrix=None, vectors=None):
        self.df = df
        self.sim_matrix = sim_matrix
        self.vectors = vectors

        items = df['item'].unique()
        self.item_map = {item: i for i, item in enumerate(items)}
        
        users = df['user'].unique()
        self.user_map = {user: i for i, user in enumerate(users)}

        if vectors is not None:
            d = vectors.shape[1]
            self.W = np.zeros((d, len(users))) # 83 x 33901
            self.b = np.zeros((1, len(users))) # 1 x 33901
            for i, user in enumerate(users):
                    id_items, ratings = get_items_rated_by_user(df, user, self.item_map)
                    rg = Ridge(alpha=0.01, fit_intercept=True)
                    Xhat = vectors[id_items, :]

                    rg.fit(Xhat, ratings)
                    self.W[:, i] = rg.coef_
                    self.b[0, i] = rg.intercept_
            
            self.Yhat = self.vectors.dot(self.W) + self.b

    def reccomend(self, user_id, n, return_result=False):
        if self.sim_matrix is not None:
            love_item = self.df[self.df['user'] == 2][['rating', 'item']].sort_values(by='rating', ascending=False)['item'].iloc[0]
            dic = self.df[love_item].sort_values(ascending=False)[1:n+1].to_dict()
            
            if return_result:
                return dic
            return list(dic.keys()) 
        
        if self.vectors is not None:
            '''
            Hãy viết theo logic kiếm những user_id sản phẩm chưa rate xong dùng self.predict để rate. 
            Xong sắp xếp giảm dần, lấy top n
            '''
            rated_items = set(self.df[self.df['user'] == user_id]['item'])
            all_items = list(self.item_map.keys())
            unrated_items = [item for item in all_items if item not in rated_items]

            item_scores = [(item, float(self.predict(item, user_id))) for item in unrated_items]

            item_scores.sort(key=lambda x: x[1], reverse=True)

            if return_result:
                return item_scores[:n]
            return [item for item, rating in item_scores[:n]]

    def predict(self, item, user_id):
        return self.Yhat[self.item_map[item], self.user_map[user_id]]

    def evaluate(self, Yhat, rates):
        se = 0
        cnt = 0
        users = self.df['user'].unique()
        for i, user in enumerate(users):
            ids, scores_truth = get_items_rated_by_user(rates, user, self.item_map)
            scores_pred = Yhat[ids, i]
            e = scores_truth - scores_pred 
            se += (e*e).sum(axis = 0)
            cnt += e.size 
        return math.sqrt(se/cnt)

# ===================================================================
# 3. MÔ HÌNH LỌC CỘNG TÁC (COLLABORATIVE FILTERING)
# ===================================================================

def knnCF(user_id, n, ratings_df, k_neighbors=8):
    train_matrix = ratings_df.pivot_table(index='user', columns='item', values='rating').fillna(0)

    item_user_matrix = train_matrix.T
    
    user_ratings = train_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index.tolist()
    unrated_items = train_matrix.columns.difference(rated_items)

    knn_model = knn(metric='cosine', algorithm='brute', n_neighbors=k_neighbors + 1)
    knn_model.fit(item_user_matrix.values)

    predictions = {}

    for item in unrated_items:
        if item not in item_user_matrix.index:
            continue

        item_idx = item_user_matrix.index.get_loc(item)
        distances, indices = knn_model.kneighbors([item_user_matrix.iloc[item_idx]], n_neighbors=k_neighbors + 1)

        neighbors = item_user_matrix.index[indices[0][1:]]
        neighbor_ratings = user_ratings[neighbors]
        neighbor_ratings = neighbor_ratings[neighbor_ratings > 0]

        if len(neighbor_ratings) > 0:
            predicted_rating = neighbor_ratings.mean()
        else:
            predicted_rating = user_ratings[user_ratings > 0].mean() if user_ratings[user_ratings > 0].any() else 2.5

        predictions[item] = predicted_rating

    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    recommended_items = [item for item, _ in sorted_preds[:n]]
    return recommended_items

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
    
    model.reccomend(user_id=2, n=10)
