import pandas as pd
import numpy as np
from src.recommender import Recommender
from src.models import iiCB, Ridge_iiCB,dcpCF
from src.utils import get_items_rated_by_user
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/captone_data.csv')
# Y_data = df[['user', 'item', 'rating']].to_numpy()
# Y_train, Y_test = train_test_split(Y_data, test_size=0.2, random_state=42)

# model = dcpCF(Y_train, K = 10, lam = .1, print_every = 10, 
#     learning_rate = 0.75, max_iter = 100, user_based = 1)
# model.fit()

# rmse = model.evaluate_RMSE(Y_test)
# print(f"RMSE test: {rmse:.3f}")

# print("Dự đoán rating:", model.pred(1390655, 'CB0103EN'))

# predicted = model.pred_for_user(1390655)
# predicted.sort(key=lambda x: x[1], reverse=True)
# print("Recommend:", predicted[:5])

# # recommender_engine = Recommender(df)
# # recommender_engine.recommend(1889878, 10, 'dcpCF')



# 1. Split train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 2. Mapping user và item trên tập train
user2idx = {user: idx for idx, user in enumerate(train_df['user'].unique())}
item2idx = {item: idx for idx, item in enumerate(train_df['item'].unique())}

# 3. Ánh xạ cho train
train_df['user_idx'] = train_df['user'].map(user2idx)
train_df['item_idx'] = train_df['item'].map(item2idx)

# 4. Ánh xạ cho test
test_df = test_df[test_df['user'].isin(user2idx.keys()) & test_df['item'].isin(item2idx.keys())].copy()
test_df['user_idx'] = test_df['user'].map(user2idx)
test_df['item_idx'] = test_df['item'].map(item2idx)

# 5. Chuyển về numpy
Y_train = train_df[['user_idx', 'item_idx', 'rating']].to_numpy()
Y_test = test_df[['user_idx', 'item_idx', 'rating']].to_numpy()

rs = dcpCF(Y_train, K=10, lam=0.1, print_every=10, learning_rate=0.75, max_iter=100, user_based=1)
rs.fit()
RMSE = rs.evaluate_RMSE(Y_test)
print('\nUser-based MF, RMSE =', RMSE)
