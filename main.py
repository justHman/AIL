import pandas as pd
from src.recommender import Recommender
from src.models import iiCB, Ridge_iiCB
from src.utils import get_items_rated_by_user

df = pd.read_csv('data\captone_data.csv')
# items , ratings = get_items_rated_by_user(df, 1889878)
# print(items)
# exit(0)
recommender_engine = Recommender(df)
model = Ridge_iiCB(df)
model.train(recommender_engine.item_vectors)
print(model.predict('CC0101EN', 1889878))
print(model.reccomend(1889878, 10, return_rating=True))
# print(rid.predict('LB0101ENv1', 2))
# print(rid.reccomend('LB0101ENv1', 10, return_rating=True))