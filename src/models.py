import pandas as pd
import numpy as np
import os 
from src.data_processing import create_similarity_matrix 
from sklearn.neighbors import NearestNeighbors as knn

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

def iiCB(
        user_id, ratings_df, item_content_df, item_similarity_matrix, 
        n=10, 
        path_simi_matrix='data\similarity_matrix\item-item.csv',
        path_data='data\captone_data.csv'
):
    """
    Hàm này làm gì:
        Gợi ý các item có nội dung (content) tương tự với những item mà
        người dùng đã thích trong quá khứ. Hữu ích cho việc gợi ý item mới
        hoặc khi dữ liệu rating còn thưa.

    Truyền vào:
        - user_id (int/str): ID của người dùng cần gợi ý.
        - ratings_df (pd.DataFrame): DataFrame chứa ['user', 'item', 'rating'].
        - item_content_df (pd.DataFrame): DataFrame nội dung item ['item', 'genre', ...].
        - item_similarity_matrix (np.array): Ma trận tương đồng giữa các item (tính trước bằng cosine_similarity).
        - n (int): Số lượng item cần gợi ý.

    Trả ra:
        - list: Danh sách các item ID được gợi ý.
    """
    if os.path.exists(path_simi_matrix):
        simi_matrix = pd.read_csv(path_simi_matrix)
    else:
        path = create_similarity_matrix(path_data, path_folder='data\similarity_matrix')
        simi_matrix = pd.read_csv(path)
    


    # Logic ví dụ: Tìm item user thích nhất -> tìm item tương tự nhất từ ma trận
    return ['item_id_A', 'item_id_B', ...]

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
