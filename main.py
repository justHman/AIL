import pandas as pd
from src.recommender import Recommender
from src.models import iiCB, Ridge_iiCB
from src.utils import get_items_rated_by_user

df = pd.read_csv('data\captone_data.csv')
recommender_engine = Recommender(df)
recommender_engine.recommend(1889878, 10, 'iiCB')
print(recommender_engine.item_vectors)