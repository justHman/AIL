import pandas as pd
from src.recommender import Recommender
from src.models import iiCB, Ridge_iiCB, knnCF
from src.utils import get_items_rated_by_user

sparse = pd.read_csv('data\captone_data.csv')
train_sparse = sparse.head(int(len(sparse) * 0.8))
test_sparse = sparse.tail(int(len(sparse) * 0.2))
recommender_engine = Recommender(train_sparse)
# model = knnCF(train_sparse, recommender_engine.iiCF_sim_matrix, recommender_engine.utility_norm)
# print(model.evaluate(train_sparse))
# print(model.evaluate(test_sparse))