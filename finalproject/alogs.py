import numpy as np
import pandas as pd

data = pd.read_table('/home/manorlf/git/RI/finalproject/data/ml-100k/u.data')
data.columns = ['user', 'movie', 'rating', 'time']


dict_user_movies = map_user_movies(data)
dict_unwatched = map_unwatched(data)
all_movies = get_all_movies(data)


def get_all_movies(data):
    all_movies = dict()
    for i in range(data.shape[0]):
        movie = data.loc[i, 'movie']
        all_movies.add(movie)
    return all_movies

def map_user_movies(data)
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
    

def map_unwatched(data):
    dict_unwatched = dict()
    for key in dict_user_movies.keys():
        dict_user_movies[key] = all_movies.difference(dict_user_movies[key])
    return dict_user_movies

def get_unwatched(user):
    return dict_unwatched[user]
