import pandas as pd
from src.recommender import Recommender
from src.models import iiCB, Ridge_iiCB
from src.utils import get_items_rated_by_user

df = pd.read_csv('data\captone_data.csv')
df_train = df.head(int(len(df) * 0.8))
df_test = df.tail(int(len(df) * 0.2))
recommender_engine = Recommender(df_train)
model = Ridge_iiCB(df_train)
model.train(recommender_engine.item_vectors)
print(model.predict('CC0101EN', 1889878))
print(model.reccomend(1889878, 10, return_rating=True))
print(model.evaluate(df_train))
print(model.evaluate(df_test))