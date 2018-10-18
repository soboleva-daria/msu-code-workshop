import pandas as pd
import numpy as np
from scipy import linalg
import math
from nltk import word_tokenize
#from nltk.corpus import stopwords

def compute_users_info(df_train,  users):
    mean_rating_users = pd.DataFrame(df_train.groupby('UserID').Rating.apply(np.mean))
    mean_rating_users.columns = ['MeanRatingUsers']
    mean_rating_users = mean_rating_users.reset_index()

    return pd.merge(
            mean_rating_users,
            users[['UserID', 'Age', 'Occupation', 'Gender']], 
            on='UserID',
            how='left'
    )

def movie_title(text): 
    title, date = text[:text.rfind('(') - 1], text[text.rfind('(') + 1 : text.rfind(')')]
    stop_words = np.load('EnglishStopwords.npy')#np.array(stopwords.words('english'))
    tokens = np.array(word_tokenize(title.lower()))
    n_tokens = float(tokens.size)
    return pd.Series([
             n_tokens,
             np.setdiff1d(tokens, stop_words).size / n_tokens,
             date
            ])

def parse_genres(x, genres):
    curr_genres = np.zeros(len(genres), dtype=int)
    curr_genres[np.in1d(genres, x.split('|'))] = 1
    return pd.Series(np.concatenate((curr_genres, [np.count_nonzero(curr_genres)])))

def mean_rating_genres(x, genres):
    mean_genres = []
    for genre in genres:
        mean_genres.append(np.mean(x[x[genre] == 1].Rating))
    return pd.Series(mean_genres)

def compute_movies_base_info(df_train, df_test, movies):
    genres = [
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western"
       ]
    columns = ['UserID', 'MovieID', 'Rating']
    movies[genres + ['n_g']] = movies.Genres.apply(lambda x: parse_genres(x, genres)) 
    df_train_new = pd.merge(df_train, movies[np.setdiff1d(movies.columns, ['Genres'])], on='MovieID', how='left')
    df_test_new = pd.merge(df_test, movies[np.setdiff1d(movies.columns, ['Genres'])], on='MovieID', how='left')
    columns.extend(genres)

    u_g = df_train_new.groupby('UserID')[genres + ['Rating']].apply(lambda x: mean_rating_genres(x, genres))
    u_g.fillna(0.0, inplace=True)
    u_g.columns = ['MeanRatingUsers{}'.format(genre) for genre in genres]
    u_g = u_g.reset_index()
    df_train_new = pd.merge(df_train_new, u_g, on='UserID', how='left')
    df_test_new = pd.merge(df_test_new, u_g, on='UserID', how='left')
     
    smooth_features = ['SmoothRatingUsers{}'.format(genre) for genre in genres]
    columns.extend(smooth_features)
    smooth = pd.DataFrame(
                    df_train_new[genres].values * df_train_new[np.setdiff1d(u_g.columns, ['UserID'])].values /
                                                            df_train_new['n_g'].values.reshape(df_train_new.shape[0], 1),
                    columns=smooth_features
                )
    df_train_new[smooth_features] = smooth
    
    smooth = pd.DataFrame(
                    df_test_new[genres].values * df_test_new[np.setdiff1d(u_g.columns, ['UserID'])].values /
                                                            df_test_new['n_g'].values.reshape(df_test_new.shape[0], 1),
                    columns=smooth_features
                )
    df_test_new[smooth_features] = smooth

    mean_rating_movies = pd.DataFrame(df_train.groupby('MovieID').Rating.apply(np.mean))
    mean_rating_movies.columns = ['MeanRatingMovies']
    mean_rating_movies = mean_rating_movies.reset_index()

    train_res = pd.merge(
                        df_train_new[columns],
                        mean_rating_movies, 
                        on='MovieID',
                        how='left'
                    ).drop(['MovieID'], axis=1)

    test_res = pd.merge(
                        df_test_new[columns],
                        mean_rating_movies, 
                        on='MovieID',
                        how='left'
                    ).fillna(0.0, axis=1).drop(['MovieID'] , axis=1)

    return train_res, test_res

def compute_movies_add_info(df_train, df_test, movies):
    genres = [
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western"
       ]
    columns = ['UserID', 'MovieID', 'Rating', 'NumTokensMovieTitle', 'FreqNonStopTokensMovieTitle', 'MovieDate']
    movies[columns[3:]] = movies.Title.apply(lambda x: movie_title(x))
    movies[genres + ['n_g']] = movies.Genres.apply(lambda x: parse_genres(x, genres)) 
    df_train_new = pd.merge(df_train, movies[np.setdiff1d(movies.columns, ['Title', 'Genres'])], on='MovieID', how='left')
    df_test_new = pd.merge(df_test, movies[np.setdiff1d(movies.columns, ['Title', 'Genres'])], on='MovieID', how='left')
    columns.extend(genres)

    u_g = df_train_new.groupby('UserID')[genres + ['Rating']].apply(lambda x: mean_rating_genres(x, genres))
    u_g.fillna(0.0, inplace=True)
    u_g.columns = ['MeanRatingUsers{}'.format(genre) for genre in genres]
    u_g = u_g.reset_index()
    df_train_new = pd.merge(df_train_new, u_g, on='UserID', how='left')
    df_test_new = pd.merge(df_test_new, u_g, on='UserID', how='left')
     
    smooth_features = ['SmoothRatingUsers{}'.format(genre) for genre in genres]
    columns.extend(smooth_features)
    smooth = pd.DataFrame(
                    df_train_new[genres].values * df_train_new[np.setdiff1d(u_g.columns, ['UserID'])].values /
                                                            df_train_new['n_g'].values.reshape(df_train_new.shape[0], 1),
                    columns=smooth_features
                )
    df_train_new[smooth_features] = smooth
    

    smooth = pd.DataFrame(
                    df_test_new[genres].values * df_test_new[np.setdiff1d(u_g.columns, ['UserID'])].values /
                                                          df_test_new['n_g'].values.reshape(df_test_new.shape[0], 1),
                    columns=smooth_features
                )
    df_test_new[smooth_features] = smooth

    mean_rating_movies = pd.DataFrame(df_train.groupby('MovieID').Rating.apply(np.mean))
    mean_rating_movies.columns = ['MeanRatingMovies']
    mean_rating_movies = mean_rating_movies.reset_index()

    df_train_new.MovieDate = df_train_new.MovieDate.astype(int) >= 1950
    #df_test_new.MovieDate = (df_test_new.MovieDate.astype(int) >= 1950).astype(int)
    mean_rating_movies_date = pd.DataFrame(df_train_new.groupby('MovieDate').Rating.apply(np.mean))
    mean_rating_movies_date.columns = ['MeanRatingMoviesDate']
    mean_rating_movies_date = mean_rating_movies_date.reset_index()

    train_res = pd.merge(
                    pd.merge(
                        df_train_new[columns],
                        mean_rating_movies, 
                        on='MovieID',
                        how='left'
                    ),
                    mean_rating_movies_date,
                    on='MovieDate',
                    how='left'
                ).drop(['MovieID', 'MovieDate'], axis=1)

    test_res = pd.merge(
                    pd.merge(
                        df_test_new[columns],
                        mean_rating_movies, 
                        on='MovieID',
                        how='left'
                    ),
                    mean_rating_movies_date,
                    on='MovieDate',
                    how='left'
                ).fillna(0.0, axis=1).drop(['MovieID', 'MovieDate'] , axis=1)

    return train_res, test_res

def binary_categorical(X_train, X_test, columns):
    for col_name in columns:
        X_train = pd.concat([X_train, pd.get_dummies(X_train[col_name])], axis=1)
        X_test = pd.concat([X_test, pd.get_dummies(X_test[col_name])], axis=1)
        X_train.drop(col_name, axis=1, inplace=True)
        X_test.drop(col_name, axis=1, inplace=True)
    return X_train, X_test

def build_categorical(X_train, X_test, movies, users, only_base_features=1):
    users = compute_users_info(X_train, users)
    if only_base_features:
        movies_train, movies_test = compute_movies_base_info(X_train, X_test, movies)
    else:
        movies_train, movies_test = compute_movies_add_info(X_train, X_test, movies)

    X_train_new = pd.merge(movies_train, users, on='UserID', how='left')
    X_test_new = pd.merge(movies_test, users, on='UserID', how='left')

    X_train_new, X_test_new = binary_categorical(X_train_new, X_test_new, ['Age', 'Occupation', 'Gender'])

    X_train_new['intercept_'] = 1
    X_test_new['intercept_'] = 1

    return X_train_new, X_test_new


class RidgeRegression(object):
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def fit(self, X, y):
        A = np.dot(X.T, X)
        Xy = np.dot(X.T, y)
        A.flat[::X.shape[1] + 1] += self.alpha
        self.w = linalg.solve(A, Xy, sym_pos=True)
        return self
        
    def predict(self, X):
        return np.dot(X, self.w)