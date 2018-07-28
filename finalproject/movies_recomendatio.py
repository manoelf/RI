from surprise import Dataset
from surprise import KNNWithMeans 
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd

data = pd.read_table('/home/manorlf/git/RI/finalproject/data/ml-100k/u.data')
data.columns = ['user', 'movie', 'rating', 'time']



def get_all_movies(data):
    all_movies = set()
    for i in range(data.shape[0]):
        movie = data.loc[i, 'movie']
        all_movies.add(movie)
    return all_movies

def map_user_movies(data):
    dict_user_movies = dict()    
    for i in range( data.shape[0]):
        user = data.loc[i, 'user']
        movie = data.loc[i, 'movie']
        if (user in dict_user_movies):
            dict_user_movies[user].add(movie)
        else:
            dict_user_movies[user] = set()
            dict_user_movies[user].add(movie)
    return dict_user_movies
    

def map_unwatched(data, all_movies):
    dict_unwatched = dict()
    for key in dict_user_movies.keys():
        dict_unwatched[key] = all_movies - dict_user_movies[key]
    return dict_unwatched

def get_unwatched(user, dict_unwatched):
    return dict_unwatched[user]


def predict_to_user(user, unwatcheds, algo):
    movies_rating = []
    for movie in unwatcheds:
        movies_rating.append(((algo.predict(uid=user, iid=str(movie)).est), movie))
    return movies_rating


def top5_recomendation(user, algo, dict_unwatched):
    unwatcheds = dict_unwatched[user]
    result = predict_to_user(user, unwatcheds, algo)
    result.sort()
    return result

dict_user_movies = map_user_movies(data)
all_movies = get_all_movies(data)
dict_unwatched = map_unwatched(data, all_movies)

#print(all_movies - dict_user_movies[245])

dataset = Dataset.load_builtin('ml-100k')

trainset, testset = train_test_split(dataset, test_size=.15)

algo = KNNWithMeans(k=40, sim_options={'name': 'cosine', 'user_based': True})

algo.fit(trainset)


print(algo.predict(uid='245', iid='2'))
print(top5_recomendation(245, algo, dict_unwatched))





