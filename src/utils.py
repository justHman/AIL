# from src.data_processing import preprocess_items
import pandas as pd
import os
import json
import numpy as np

def get_items_rated_by_user(sparse, user):
    user_rated = sparse[sparse['user'] == user][['item', 'rating']]
    items = user_rated['item'].to_numpy()
    ratings = user_rated['rating'].to_numpy()

    return items, ratings

def load_config():
    # Lấy đường dẫn tuyệt đối đến thư mục gốc project
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Tạo đường dẫn đến file config
    config_path = os.path.join(root_dir, "config", "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config

def load_sim_matrix():
    path_sim_matrix = 'data\similarity_matrix\item-item.csv'
    if os.path.exists(path_sim_matrix):
        return pd.read_csv(path_sim_matrix)
    
    path_data = 'data\captone_data.csv'

if __name__ == '__main__':
    sparse = pd.read_csv('data\captone_data.csv')
    user = 2
    get_items_rated_by_user(sparse, user)

