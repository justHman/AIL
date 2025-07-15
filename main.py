import pandas as pd
from src.recommender import Recommender
from src.models import iiCB

df = pd.read_csv('data\captone_data.csv')
recommender_engine = Recommender(df)

# df2 = pd.read_csv('data\captone_data.csv')
model = iiCB(df, vectors=recommender_engine.vector_items)

recs = model.reccomend(user_id=1889878, n=10, return_result=False)
print(recs)

exit(0)

user_1 = 1889878 # Một user đã có trong hệ thống
recs_for_user_1 = recommender_engine.get_recommendations(user_1, n=5)
print(f"Gợi ý cho người dùng cũ {user_1}:")
print(recs_for_user_1)

# Kịch bản 1: Người dùng cũ
user_1 = 1889878 # Một user đã có trong hệ thống
recs_for_user_1 = recommender_engine.get_recommendations(user_1, n=5)
print(f"Gợi ý cho người dùng cũ {user_1}:")
print(recs_for_user_1)


# Kịch bản 3: Gợi ý một item mới cho ai
new_item = 'NEW0101EN' # Một item mới (giả sử đã có trong items.csv)
users_to_target = recommender_engine.recommend_for_new_item(new_item, n_users=5)
print(f"\nNên gợi ý item mới '{new_item}' cho những người dùng sau:")
print(users_to_target)

items_to_target = recommender_engine.recommend_for_new_item(new_item, n_users=5)