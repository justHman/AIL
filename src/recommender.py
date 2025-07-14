import pandas as pd
from . import models # Import các mô hình từ file models.py

class HybridRecommender:
    def __init__(self, ratings_df, items_df):
        """
        Hàm này làm gì:
            Khởi tạo hệ thống gợi ý. Tại đây sẽ tải dữ liệu và tính toán
            trước các thành phần cần thiết (như các ma trận tương đồng,
            huấn luyện mô hình SVD) để tăng tốc độ khi gợi ý.

        Truyền vào:
            - ratings_df (pd.DataFrame): Dữ liệu rating.
            - items_df (pd.DataFrame): Dữ liệu nội dung item.
        """
        self.ratings_df = ratings_df
        self.items_df = items_df
        self.user_ids = ratings_df['user'].unique()
        self.item_ids = ratings_df['item'].unique()

        # Các thành phần được tính toán trước
        self.svd_predicted_ratings = None # Sẽ được tính khi init
        self.content_similarity_matrix = None # Sẽ được tính khi init
        self.cf_item_similarity_matrix = None # Sẽ được tính khi init

        print("Recommender is ready.")


    def get_recommendations(self, user_id, n=10):
        """
        Hàm này làm gì:
            Hàm gợi ý chính. Tự động xác định kịch bản (người dùng mới/cũ)
            và áp dụng chiến lược gợi ý phù hợp.

        Truyền vào:
            - user_id (int/str): ID của người dùng.
            - n (int): Số lượng gợi ý mong muốn.

        Trả ra:
            - dict: Một dictionary chứa danh sách gợi ý và chiến lược đã dùng.
                    Ví dụ: {'recommendations': [...], 'strategy': 'cold_start_popular'}
        """
        # === KỊCH BẢN 1: NGƯỜI DÙNG MỚI (COLD-START) ===
        if user_id not in self.user_ids:
            print(f"User {user_id} is a new user. Applying cold-start strategy.")
            recs = models.get_popular_items(self.ratings_df, top_n=n)
            return {'recommendations': recs, 'strategy': 'cold_start_popular'}

        # === KỊCH BẢN 2: NGƯỜI DÙNG CŨ ===
        print(f"User {user_id} is an existing user. Applying hybrid strategy.")
        
        # Lấy danh sách item user đã xem để loại bỏ
        items_rated = self.ratings_df[self.ratings_df['user'] == user_id]['item'].tolist()

        # Lấy gợi ý từ nhiều mô hình khác nhau
        svd_recs = models.get_matrix_factorization_recs(
            user_id, self.svd_predicted_ratings, items_rated, n
        )
        item_cf_recs = models.get_item_item_recs(
            user_id, self.ratings_df, self.cf_item_similarity_matrix, n
        )

        # Logic kết hợp (hybrid): ví dụ, kết hợp và xếp hạng lại 2 danh sách trên
        hybrid_recs = list(dict.fromkeys(svd_recs + item_cf_recs))[:n]

        return {'recommendations': hybrid_recs, 'strategy': 'hybrid_svd_item_cf'}


    def recommend_for_new_item(self, new_item_id, n_users=10):
        """
        Hàm này làm gì:
            Giải quyết vấn đề "item cold-start". Tìm ra những người dùng
            phù hợp nhất để gợi ý một item hoàn toàn mới.

        Truyền vào:
            - new_item_id (str): ID của item mới.
            - n_users (int): Số lượng người dùng cần tìm để gợi ý.

        Trả ra:
            - list: Danh sách các user ID là ứng viên tốt nhất để nhận gợi ý
                    về item mới này.
        """
        # Logic:
        # 1. Dùng Content-Based để tìm các item cũ tương tự nhất với `new_item_id`.
        # 2. Tìm những người dùng đã đánh giá cao các item tương tự đó.
        # 3. Trả về danh sách những người dùng này.
        
        if new_item_id not in self.items_df['item'].tolist():
             return {'error': 'Item content not found.'}
    
        # Tìm các users phù hợp dựa trên content
        # ... (logic chi tiết sẽ được cài đặt ở đây) ...
        
        return ['user_id_101', 'user_id_202', ...]