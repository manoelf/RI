from surprise import Dataset
from surprise import KNNWithMeans 
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split
import pandas as pd

data = pd.read_table('/home/manorlf/git/RI/finalproject/data/ml-100k/u.data')
data.columns = ['user', 'movie', 'rating', 'time']
movies_data = open('/home/manorlf/git/RI/finalproject/data/ml-100k/u.item', encoding='ISO-8859-1')

def map_movie_id_name(movies_data):
    dict_movies = dict()
    for line in movies_data:
        item = line.split('|')
        movie_id = item[0]
        movie_name = item[1]
        dict_movies[movie_id] = movie_name
    return dict_movies


dict_movies = map_movie_id_name(movies_data)

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
        pred_rating = algo.predict(uid=str(user), iid=str(movie)).est
        movies_rating.append((pred_rating, movie))
    return movies_rating


def top5_recomendation(user, algo, dict_unwatched):
    unwatcheds = dict_unwatched[user]
    result = predict_to_user(user, unwatcheds, algo)
    result.sort()
    return result[-5:]

dict_user_movies = map_user_movies(data)
all_movies = get_all_movies(data)
dict_unwatched = map_unwatched(data, all_movies)

#print(all_movies - dict_user_movies[245])

dataset = Dataset.load_builtin('ml-100k')

trainset, testset = train_test_split(dataset, test_size=.15)

algo = KNNWithMeans(k=40, sim_options={'name': 'cosine', 'user_based': True})

algo.fit(trainset)


print(algo.predict(uid='245', iid='2'))
top5 = top5_recomendation(245, algo, dict_unwatched)

print('Movies watched:')
for i in dict_user_movies[245]:
    print(dict_movies[str(i)])

print('Top reconmendation:')
for i in top5:
    print(dict_movies[str(i[1])])





