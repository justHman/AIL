import pandas as pd
from src.recommender import Recommender
from src.models import iiCB, Ridge_iiCB

df = pd.read_csv('data\captone_data.csv')
recommender_engine = Recommender(df)
print(recommender_engine.item_vectors)
model = Ridge_iiCB(df)
print(model.item_map)
# print(rid.predict('LB0101ENv1', 2))
# print(rid.reccomend('LB0101ENv1', 10, return_rating=True))