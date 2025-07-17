import pandas as pd
import numpy as np
import os 
import math
from sklearn.neighbors import NearestNeighbors
from src.data_processing import create_item_similarity_matrix 
from src.utils import get_items_rated_by_user
from sklearn.linear_model import Ridge

# ===================================================================
# 1. MÔ HÌNH PHI CÁ NHÂN HÓA (DÙNG CHO NGƯỜI DÙNG MỚI)
# ===================================================================

def get_popular_items(ratings_df, top_n=10, min_ratings=5):
    """
    Hàm này làm gì:
        Lấy ra danh sách các item phổ biến nhất dựa trên số lượng rating hoặc
        điểm rating trung bình cao nhất. Rất hữu ích cho người dùng mới.

    Truyền vào:
        - ratings_df (pd.DataFrame): DataFrame chứa ['user', 'item', 'rating'].
        - top_n (int): Số lượng item cần gợi ý.
        - min_ratings (int): Số lượng rating tối thiểu để một item được xem xét.

    Trả ra:
        - list: Một danh sách chứa các item ID phổ biến nhất.
    """
    item_stats = ratings_df.groupby('item').agg(
        count=('rating', 'count'),
        avg_rating=('rating', 'mean')
    )
    
    popular_items = item_stats[item_stats['count'] >= min_ratings]
    popular_items = popular_items.sort_values(by=['avg_rating', 'count'], ascending=[False, False]).head(top_n)
    print(popular_items)
    return popular_items.index.tolist()

    # Logic ví dụ: Đếm số lần xuất hiện của mỗi item
    # item_counts = ratings_df['item'].value_counts()
    # ... lọc và trả về top N ...

# ===================================================================
# 2. MÔ HÌNH LỌC DỰA TRÊN NỘI DUNG (CONTENT-BASED / ITEM-ITEM) 
# ===================================================================
class Ridge_iiCB:
    def __init__(self, df):
        self.df = df

        self.items = df['item'].unique()
        self.item_map = {item: i for i, item in enumerate(self.items)}
        
        self.users = df['user'].unique()
        self.user_map = {user: i for i, user in enumerate(self.users)}
        
        self.X = None
        self.W = None
        self.b = None            
        self.y = None

    def train(self, vectors):
        self.X = vectors
        self.W = np.zeros((vectors.shape[1], len(self.users))) # 83 x 33901
        self.b = np.zeros((1, len(self.users)))                # 1  x 33901

        for i, user in enumerate(self.users):
            id_items, ratings = get_items_rated_by_user(df, user, self.item_map)
            X_train = self.X[id_items, :]

            rg = Ridge(alpha=0.01, fit_intercept=True)
            rg.fit(X_train, ratings)
            
            self.W[:, i] = rg.coef_
            self.b[0, i] = rg.intercept_
            
        self.y = self.X @ self.W + self.b

    def predict(self, item, user_id):
        return self.y[self.item_map[item], self.user_map[user_id]]

    def recomend(self, user_id, n, return_rating=False):
        rated_items = set(self.df[self.df['user'] == user_id]['item'])
        all_items = list(self.item_map.keys())
        unrated_items = [item for item in all_items if item not in rated_items]

        item_scores = [(item, float(self.predict(item, user_id))) for item in unrated_items]

        item_scores.sort(key=lambda x: x[1], reverse=True)

        if return_rating:
            return item_scores[:n]
        return [item for item, rating in item_scores[:n]]
    
    def evaluate(self, df):
        se = 0
        cnt = 0
        users = self.df['user'].unique()
        for i, user in enumerate(users):
            id_items, ratings = get_items_rated_by_user(df, user, self.item_map)
            ratings_pred = self.y[id_items, i]
            e = ratings - ratings_pred 
            se += (e*e).sum(axis = 0)
            cnt += e.size 
        return math.sqrt(se/cnt)

# ===================================================================
# 3. MÔ HÌNH LỌC CỘNG TÁC (COLLABORATIVE FILTERING)
# ===================================================================

def get_item_item_recs(user_id, ratings_df, item_similarity_matrix, n=10, k_neighbors=8):
    """
    Hàm này làm gì:
        Thực hiện Lọc cộng tác Item-Item. Gợi ý các item tương tự với những
        item người dùng đã đánh giá cao, dựa trên hành vi của toàn bộ cộng đồng.

    Truyền vào:
        - user_id (int/str): ID của người dùng cần gợi ý.
        - ratings_df (pd.DataFrame): DataFrame chứa ['user', 'item', 'rating'].
        - item_similarity_matrix (np.array): Ma trận tương đồng item-item (tính từ ma trận user-item).
        - n (int): Số lượng item cần gợi ý.

    Trả ra:
        - list: Danh sách các item ID được gợi ý.
    """
    train_matrix = ratings_df.pivot_table(index='user', columns='item', values='rating').fillna(0)

    if user_id not in train_matrix.index:
        print(f"User '{user_id}' không tồn tại trong dữ liệu.")
        return []

    item_user_matrix = train_matrix.T
    
    user_ratings = train_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index.tolist()
    unrated_items = train_matrix.columns.difference(rated_items)

    knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k_neighbors + 1)
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


def get_matrix_factorization_recs(user_id, predicted_ratings_df, items_rated_by_user, n=10):
    """
    Hàm này làm gì:
        Sử dụng kết quả từ một mô hình Phân rã Ma trận (như SVD) để đưa ra gợi ý.
        Đây là phương pháp Lọc cộng tác hiện đại và hiệu quả.

    Truyền vào:
        - user_id (int/str): ID của người dùng cần gợi ý.
        - predicted_ratings_df (pd.DataFrame): Ma trận user-item đã được điền đầy
          bằng các rating dự đoán từ SVD.
        - items_rated_by_user (list): Danh sách các item người dùng đã xem để loại bỏ.
        - n (int): Số lượng item cần gợi ý.

    Trả ra:
        - list: Danh sách các item ID được gợi ý.
    """
    # Logic ví dụ: Sắp xếp các rating dự đoán cho user, loại bỏ item đã xem
    if user_id not in predicted_ratings_df.index:
        print(f"User '{user_id}' không tồn tại trong dữ liệu dự đoán.")
        return []

    # Lấy toàn bộ rating dự đoán của user đó
    user_predicted_ratings = predicted_ratings_df.loc[user_id]

    # Loại bỏ các item mà user đã đánh giá
    filtered_ratings = user_predicted_ratings.drop(labels=items_rated_by_user, errors='ignore')

    # Sắp xếp theo điểm dự đoán giảm dần
    recommended_items = filtered_ratings.sort_values(ascending=False).head(n).index.tolist()

    return recommended_items