import pandas as pd
import numpy as np
import os 
from src.data_processing import create_item_similarity_matrix 

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
    # Logic ví dụ: Đếm số lần xuất hiện của mỗi item
    # item_counts = ratings_df['item'].value_counts()
    # ... lọc và trả về top N ...
    return ['item_id_1', 'item_id_2', ...]

# ===================================================================
# 2. MÔ HÌNH LỌC DỰA TRÊN NỘI DUNG (CONTENT-BASED / ITEM-ITEM) 
# ===================================================================

def get_content_based_recs(
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
        path = create_item_similarity_matrix(path_data, path_folder='data\similarity_matrix')
        simi_matrix = pd.read_csv(path)
    


    # Logic ví dụ: Tìm item user thích nhất -> tìm item tương tự nhất từ ma trận
    return ['item_id_A', 'item_id_B', ...]

# ===================================================================
# 3. MÔ HÌNH LỌC CỘNG TÁC (COLLABORATIVE FILTERING)
# ===================================================================

def get_item_item_recs(user_id, ratings_df, item_similarity_matrix, n=10):
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
    return ['item_id_X', 'item_id_Y', ...]


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
    return ['item_id_SVD1', 'item_id_SVD2', ...]