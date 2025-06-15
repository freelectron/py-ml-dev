import pandas as pd


ratings = pd.read_csv("../../static/data/movielens/ml-100k/u.data", delimiter='\t', header=None, names=['user','movie','rating','timestamp'])
movies = pd.read_csv("../../static/data/movielens/ml-100k/u.item",  delimiter='|', encoding='latin-1',
                     usecols=(0,1), names=('movie','title'), header=None)
users = pd.read_csv("../../static/data/movielens/ml-100k/u.user", delimiter="|", header=None, usecols=(0,1,2), names=("user", "age", "gender"))

ratings_original = ratings.copy(deep=True)
ratings = ratings.merge(movies, on='movie').merge(users, on="users")

