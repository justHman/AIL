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
from scipy import sparse 
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse 
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
    
    def evaluate(self, df):
        if self.y is None:
            return
        
        truth = []
        predict = []
        for user in self.users:
            items, ratings = get_items_rated_by_user(df, user)
            ratings_pred = self.predict(items, user)
            truth.extend(ratings)
            predict.extend(ratings_pred)

        mae = mean_absolute_error(truth, predict)
        mse = mean_squared_error(truth, predict)
        r2 = r2_score(truth, predict)
        rmse = np.sqrt(mse)

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

# ===================================================================
# 4. PHÂN RÃ MA TRẬN (MATRIX FATORIZATION COLLABORATIVE FILTERING)
# ===================================================================

# def dcpCF(user_id, n):
    
#     """
#     Hàm này làm gì:
#         Sử dụng phương pháp phân rã ma trận - là 1 trong 2 phương pháp của Colabtive Filltering.
#         Sử dụng kết quả từ một mô hình Phân rã Ma trận (như SVD) để đưa ra gợi ý.
#         Đây là phương pháp Lọc cộng tác hiện đại và hiệu quả.

#     Input:
#         - user_id (int/str): ID của người dùng cần gợi ý.
#         - n (int): Số lượng item cần gợi ý.
#         - Có thể modify thêm biến đề hoàn thành hàm này (vẫn giữ nguyên user_id và n)

#     Output:
#         - list: Danh sách các item ID được gợi ý.
#     """
#     # Logic ví dụ: Sắp xếp các rating dự đoán cho user, loại bỏ item đã xem
#     return ['item_id_SVD1', 'item_id_SVD2', ...]

# class dcpCF(object):
#     """docstring for CF"""
#     def __init__(self, Y_data, K, lam = 0.1, Xinit = None, Winit = None, 
#             learning_rate = 0.5, max_iter = 1000, print_every = 100, user_based = 1):
#         self.Y_raw_data = Y_data
#         self.K = K
#         # regularization parameter
#         self.lam = lam
#         # learning rate for gradient descent
#         self.learning_rate = learning_rate
#         # maximum number of iterations
#         self.max_iter = max_iter
#         # print results after print_every iterations
#         self.print_every = print_every
#         # user-based or item-based
#         self.user_based = user_based
#         # number of users, items, and ratings. Remember to add 1 since id starts from 0
#         self.n_users = int(np.max(Y_data[:, 0])) + 1 
#         self.n_items = int(np.max(Y_data[:, 1])) + 1
#         self.n_ratings = Y_data.shape[0]
        
#         if Xinit is None: # new
#             self.X = np.random.randn(self.n_items, K)
#         else: # or from saved data
#             self.X = Xinit 
        
#         if Winit is None: 
#             self.W = np.random.randn(K, self.n_users)
#         else: # from daved data
#             self.W = Winit
            
#         # normalized data, update later in normalized_Y function
#         self.Y_data_n = self.Y_raw_data.copy()


#     def normalize_Y(self):
#         if self.user_based:
#             user_col = 0
#             item_col = 1
#             n_objects = self.n_users

#         # if we want to normalize based on item, just switch first two columns of data
#         else: # item bas
#             user_col = 1
#             item_col = 0 
#             n_objects = self.n_items

#         users = self.Y_raw_data[:, user_col] 
#         self.mu = np.zeros((n_objects,))
#         for n in range(n_objects):
#             # row indices of rating done by user n
#             # since indices need to be integers, we need to convert
#             ids = np.where(users == n)[0].astype(np.int32)
#             # indices of all ratings associated with user n
#             item_ids = self.Y_data_n[ids, item_col] 
#             # and the corresponding ratings 
#             ratings = self.Y_data_n[ids, 2]
#             # take mean
#             m = np.mean(ratings) 
#             if np.isnan(m):
#                 m = 0 # to avoid empty array and nan value
#             self.mu[n] = m
#             # normalize
#             self.Y_data_n[ids, 2] = ratings - self.mu[n]
#     def loss(self):
#         L = 0 
#         for i in range(self.n_ratings):
#             # user, item, rating
#             n, m, rate = int(self.Y_data_n[i, 0]), int(self.Y_data_n[i, 1]), self.Y_data_n[i, 2]
#             L += 0.5*(rate - self.X[m, :].dot(self.W[:, n]))**2
        
#         # take average
#         L /= self.n_ratings
#         # regularization, don't ever forget this 
#         L += 0.5*self.lam*(np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))
#         return L 
#     def get_items_rated_by_user(self, user_id):
#         """
#         get all items which are rated by user user_id, and the corresponding ratings
#         """
#         ids = np.where(self.Y_data_n[:,0] == user_id)[0] 
#         item_ids = self.Y_data_n[ids, 1].astype(np.int32) # indices need to be integers
#         ratings = self.Y_data_n[ids, 2]
#         return (item_ids, ratings)
        
        
#     def get_users_who_rate_item(self, item_id):
#         """
#         get all users who rated item item_id and get the corresponding ratings
#         """
#         ids = np.where(self.Y_data_n[:,1] == item_id)[0] 
#         user_ids = self.Y_data_n[ids, 0].astype(np.int32)
#         ratings = self.Y_data_n[ids, 2]
#         return (user_ids, ratings)
#     def updateX(self):
#         for m in range(self.n_items):
#             user_ids, ratings = self.get_users_who_rate_item(m)
#             Wm = self.W[:, user_ids]
#             # gradient
#             grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T)/self.n_ratings + \
#                                                self.lam*self.X[m, :]
#             self.X[m, :] -= self.learning_rate*grad_xm.reshape((self.K,))
    
#     def updateW(self):
#         for n in range(self.n_users):
#             item_ids, ratings = self.get_items_rated_by_user(n)
#             Xn = self.X[item_ids, :]
#             # gradient
#             grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n]))/self.n_ratings + \
#                         self.lam*self.W[:, n]
#             self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.K,))   
#     def fit(self):
#         self.normalize_Y()
#         for it in range(self.max_iter):
#             self.updateX()
#             self.updateW()
#             if (it + 1) % self.print_every == 0:
#                 rmse_train = self.evaluate_RMSE(self.Y_raw_data)
#                 print (f'iter = ,{it + 1}, loss = {self.loss()},  RMSE train ={rmse_train}')
    
#     def pred(self, u, i):
#         """ 
#         predict the rating of user u for item i 
#         if you need the un
#         """
#         u = int(u)
#         i = int(i)
#         if self.user_based:
#             bias = self.mu[u]
#         else: 
#             bias = self.mu[i]
#         pred = self.X[i, :].dot(self.W[:, u]) + bias 
#         # truncate if results are out of range [0, 5]
#         if pred < 0:
#             return 0 
#         if pred > 5: 
#             return 5 
#         return pred 
        
    
#     def pred_for_user(self, user_id):
#         """
#         predict ratings one user give all unrated items
#         """
#         ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
#         items_rated_by_u = self.Y_data_n[ids, 1].tolist()              
        
#         y_pred = self.X.dot(self.W[:, user_id]) + self.mu[user_id]
#         predicted_ratings= []
#         for i in range(self.n_items):
#             if i not in items_rated_by_u:
#                 predicted_ratings.append((i, y_pred[i]))
        
#         return predicted_ratings
#     def evaluate_RMSE(self, rate_test):
#         n_tests = rate_test.shape[0]
#         SE = 0 # squared error
#         for n in range(n_tests):
#             pred = self.pred(rate_test[n, 0], rate_test[n, 1])
#             SE += (pred - rate_test[n, 2])**2 

#         RMSE = np.sqrt(SE/n_tests)
#         return RMSE
    


class dcpCF(object):
    def __init__(self, Y_data, K, lam = 0.1, Xinit = None, Winit = None, 
                 learning_rate = 0.5, max_iter = 1000, print_every = 100, user_based = 1):
        # Xử lý dữ liệu: Tự động mã hóa user, item
        users = pd.Series(Y_data[:, 0])
        items = pd.Series(Y_data[:, 1])
        self.user2idx = {k: v for v, k in enumerate(users.unique())}
        self.item2idx = {k: v for v, k in enumerate(items.unique())}
        self.idx2user = {v: k for k, v in self.user2idx.items()}
        self.idx2item = {v: k for k, v in self.item2idx.items()}
        
        # Map dữ liệu sang số nguyên
        Y_data_idx = np.array([
            [self.user2idx[u], self.item2idx[i], float(r)] for u, i, r in Y_data
        ])
        self.Y_raw_data = Y_data_idx
        self.K = K
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.print_every = print_every
        self.user_based = user_based
        self.n_users = len(self.user2idx)
        self.n_items = len(self.item2idx)
        self.n_ratings = Y_data.shape[0]
        
        if Xinit is None:
            self.X = np.random.randn(self.n_items, K)
        else:
            self.X = Xinit
        if Winit is None:
            self.W = np.random.randn(K, self.n_users)
        else:
            self.W = Winit
        
        self.Y_data_n = self.Y_raw_data.copy()
        self.mu = None

    def _get_user_idx(self, u):
        if isinstance(u, int) and (0 <= u < self.n_users):
            return u
        return self.user2idx[u]

    def _get_item_idx(self, i):
        if isinstance(i, int) and (0 <= i < self.n_items):
            return i
        return self.item2idx[i]

    def normalize_Y(self):
        if self.user_based:
            user_col = 0
            item_col = 1
            n_objects = self.n_users
        else:
            user_col = 1
            item_col = 0
            n_objects = self.n_items

        users = self.Y_raw_data[:, user_col]
        self.mu = np.zeros((n_objects,))
        for n in range(n_objects):
            ids = np.where(users == n)[0].astype(np.int32)
            ratings = self.Y_data_n[ids, 2]
            m = np.mean(ratings) 
            if np.isnan(m):
                m = 0
            self.mu[n] = m
            self.Y_data_n[ids, 2] = ratings - self.mu[n]

    def loss(self):
        L = 0 
        for i in range(self.n_ratings):
            n, m, rate = int(self.Y_data_n[i, 0]), int(self.Y_data_n[i, 1]), self.Y_data_n[i, 2]
            L += 0.5 * (rate - self.X[m, :].dot(self.W[:, n]))**2
        L /= self.n_ratings
        L += 0.5 * self.lam * (np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro'))
        return L

    def get_items_rated_by_user(self, user_id):
        user_idx = self._get_user_idx(user_id)
        ids = np.where(self.Y_data_n[:, 0] == user_idx)[0] 
        item_ids = self.Y_data_n[ids, 1].astype(np.int32)
        ratings = self.Y_data_n[ids, 2]
        return (item_ids, ratings)
        
    def get_users_who_rate_item(self, item_id):
        item_idx = self._get_item_idx(item_id)
        ids = np.where(self.Y_data_n[:, 1] == item_idx)[0] 
        user_ids = self.Y_data_n[ids, 0].astype(np.int32)
        ratings = self.Y_data_n[ids, 2]
        return (user_ids, ratings)

    def updateX(self):
        for m in range(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m)
            Wm = self.W[:, user_ids]
            grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T)/self.n_ratings + self.lam * self.X[m, :]
            self.X[m, :] -= self.learning_rate * grad_xm.reshape((self.K,))
    
    def updateW(self):
        for n in range(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n]))/self.n_ratings + self.lam * self.W[:, n]
            self.W[:, n] -= self.learning_rate * grad_wn.reshape((self.K,))

    def fit(self):
        self.normalize_Y()
        for it in range(self.max_iter):
            self.updateX()
            self.updateW()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y_raw_data)
                print(f'iter = {it + 1}, loss = {self.loss():.4f},  RMSE train = {rmse_train:.4f}')

    def pred(self, u, i):
        u_idx = self._get_user_idx(u)
        i_idx = self._get_item_idx(i)
        bias = self.mu[u_idx] if self.user_based else self.mu[i_idx]
        pred = self.X[i_idx, :].dot(self.W[:, u_idx]) + bias
        return max(0, min(5, pred))

    def pred_for_user(self, user_id):
        u_idx = self._get_user_idx(user_id)
        items_rated_by_u = set(self.Y_data_n[self.Y_data_n[:, 0] == u_idx, 1])
        y_pred = self.X.dot(self.W[:, u_idx]) + self.mu[u_idx]
        predicted_ratings = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                predicted_ratings.append((self.idx2item[i], y_pred[i]))
        return predicted_ratings

    def evaluate_RMSE(self, rate_test):
        # rate_test: có thể là raw (user gốc, item gốc, rating) hoặc index
        # Tự động xử lý luôn!
        if isinstance(rate_test, np.ndarray) and rate_test.shape[1] == 3:
            if not np.issubdtype(rate_test[:,0].dtype, np.integer):
                # Nếu là user/item gốc (string, số lạ), cần encode về index
                rate_test_idx = np.array([
                    [self._get_user_idx(u), self._get_item_idx(i), float(r)] for u, i, r in rate_test
                ])
            else:
                rate_test_idx = rate_test
        else:
            raise ValueError("rate_test phải là numpy array với 3 cột (user, item, rating)")
        n_tests = rate_test_idx.shape[0]
        SE = 0
        for n in range(n_tests):
            pred = self.pred(int(rate_test_idx[n, 0]), int(rate_test_idx[n, 1]))
            SE += (pred - rate_test_idx[n, 2])**2
        RMSE = np.sqrt(SE / n_tests)
        return RMSE

if __name__ == '__main__':
    df = pd.read_csv('data\oversampling_data.csv')
    model = iiCB(df.tail(100), vectors=True)
    
    model.recommend(user_id=2, n=10)
