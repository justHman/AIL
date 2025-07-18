import pandas as pd
from src.recommender import Recommender
from src.models import iiCB, Ridge_iiCB
from src.utils import get_items_rated_by_user

df = pd.read_csv('data\captone_data.csv')
recommender_engine = Recommender(df)
recs = recommender_engine.recommend(user=2, n=10, aglorithm='knnCF')
print(recs)