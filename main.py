# File: main.py
import pandas as pd
from src.recommender import HybridRecommender

# 1. Tải dữ liệu
ratings_data = pd.read_csv('data\oversampling_data.csv')
items_data = pd.read_csv('data\oversampling_data.csv')

# 2. Khởi tạo hệ thống (bước này có thể mất thời gian để huấn luyện)
recommender_engine = HybridRecommender(ratings_data, items_data)

# 3. Lấy gợi ý cho các kịch bản khác nhau

# Kịch bản 1: Người dùng cũ
user_1 = 1889878 # Một user đã có trong hệ thống
recs_for_user_1 = recommender_engine.get_recommendations(user_1, n=5)
print(f"Gợi ý cho người dùng cũ {user_1}:")
print(recs_for_user_1)

# Kịch bản 2: Người dùng mới
user_2 = 9999999 # Một user chưa từng có
recs_for_user_2 = recommender_engine.get_recommendations(user_2, n=5)
print(f"\nGợi ý cho người dùng mới {user_2}:")
print(recs_for_user_2)


# Kịch bản 3: Gợi ý một item mới cho ai
new_item = 'NEW0101EN' # Một item mới (giả sử đã có trong items.csv)
users_to_target = recommender_engine.recommend_for_new_item(new_item, n_users=5)
print(f"\nNên gợi ý item mới '{new_item}' cho những người dùng sau:")
print(users_to_target)