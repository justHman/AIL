import pandas as pd
from src.recommender import Recommender
from src.models import iiCB, Ridge_iiCB
from src.utils import get_items_rated_by_user

df = pd.read_csv('data\captone_data.csv')
recommender_engine = Recommender(df)
model = Ridge_iiCB(df)
model.train(recommender_engine.item_vectors)
print(model.predict('CC0101EN', 1889878))
print(model.reccomend(1889878, 10, return_rating=True))