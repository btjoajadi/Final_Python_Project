#packages & libraries
from spotipy.oauth2 import SpotifyClientCredentials 
import pandas as pd #Dataframe, Series
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import io
from scipy import misc
import spotipy
import spotipy.util as util
sp = spotipy.Spotify() 
import csv
import random
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

##################################################################################
# Data Pre-Processing
##################################################################################

'''
Making a new column called "TARGET" that classifies the good and bad playlists 

good : the individual's playlist
bad: playlist of songs that the individual dislikes 

'''
def good_playlist(good_playlist_csv):
    good = pd.read_csv(good_playlist_csv)
    target = 1
    good['target'] = target
    return good


def bad_playlist(bad_playlist_csv):
    bad = pd.read_csv(bad_playlist_csv)
    target = 0
    bad['target'] = target
    return bad


def heatMap(df):
    '''
    heatmaps function checking for any missing values in the dataframe
    '''
    corr = df.corr()   
    fig, ax = plt.subplots(figsize=(10, 10))     
    colormap = sns.diverging_palette(220, 10, as_cmap=True)   
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")   
    plt.xticks(range(len(corr.columns)),   
    corr.columns);     
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()   


def good_hist(df):
    '''
    Histograms comparing the variables in each playlist
    '''
    good = good_playlist(df)
    good.hist(alpha=0.7, bins=30, label='positive')
    plt.legend(loc='upper right')
    return plt.show()


def bad_hist(df):
    '''
    Histograms comparing the variables in each playlist
    '''
    bad = bad_playlist(df)
    bad.hist(alpha=0.7, bins=30, label='negative')
    plt.legend(loc='upper right')
    return plt.show()


def one_playlist(df_1, df_2):
    '''
    Combining the good and bad playlist into one dataframe
    '''
    good = good_playlist(df_1)
    bad = bad_playlist(df_2)
    frames = [good, bad]
    combined = pd.concat(frames)
    return combined


def remove_col(df_1, df_2):
    '''
    Deleting the unnecessary columns that label row IDs
    '''
    combined = one_playlist(df_1, df_2)
    keep_col = ['artist_name', 'track_name', 'track_id', 'popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'target']
    new_rr = combined[keep_col]
    return new_rr


def variable_type(df_1, df_2):
    '''
    Changing Variable Types from object to numeric values 
    '''
    variable = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness","duration_ms"]
    for audio_feature in variable:
        new_rr = remove_col(df_1, df_2)
        new_rr[audio_feature] = pd.to_numeric(new_rr[audio_feature],errors='coerce')
    return new_rr.dtypes


##################################################################################
# Building Prediction Models (Decision Tree Classification, Random Forest Classifier, KNN Classifier)
##################################################################################


# df_1 = 'Carmen_Spotify.csv' # THIS NEEDS TO BE CHANGED FOR DIFFERENT PLAYLISTS
# df_2 = 'BadList_Spotify.csv' # THIS NEEDS TO BE CHANGED FOR DIFFERENT PLAYLISTS

new_rr = remove_col(df_1, df_2)
random_seed = 5 #set random seed for reproducible results 
variables = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness","duration_ms"]
X = new_rr[variables] #using the variables we would like to use 
y = new_rr["target"] #target variable 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed) # 80% training and 20% test


def decision_tree():
    # Decision Tree Classification Model
    first_DT_clf = DecisionTreeClassifier() # Decision Tree classifer object
    first_DT_clf = first_DT_clf.fit(X_train, y_train) # Train Decision Tree Classifer
    y_pred = first_DT_clf.predict(X_test) #Predict the response for test dataset

    # Decision Tree Model Accuracy
    accuracy = (accuracy_score(y_test, y_pred))
    print(f'Accuracy: {accuracy*100}%') #accuracy

    # Decision Tree Classifier Confusion Matrix
    results = confusion_matrix(y_test, y_pred) 
    print('Confusion Matrix :')
    print(results) 
    print('Report for Decision Tree Model : ')
    print(classification_report(y_test, y_pred))


def random_forest():
    # Random Forest Tree Model
    RF_CLF = RandomForestClassifier()
    RF_CLF.fit(X_train, y_train)
    RF_pred = RF_CLF.predict(X_test)

    # Random Forest Model Accuracy
    accuracy_RF = (accuracy_score(y_test, RF_pred))
    print(f'Accuracy: {accuracy_RF*100}%')

    # Random Forest Tree Model Confusion Matrix 
    results = confusion_matrix(y_test, RF_pred) 
    print('Confusion Matrix :')
    print(results) 
    print('Report for Random Forest Model : ')
    print(classification_report(y_test, RF_pred))


def knn_model():
    #KNN Model
    knn = KNeighborsClassifier(3)
    knn.fit(X_train, y_train)
    first_DT_clf = DecisionTreeClassifier() # Decision Tree classifer object
    first_DT_clf = first_DT_clf.fit(X_train, y_train) # Train Decision Tree Classifer
    knn_pred = first_DT_clf.predict(X_test)

    # KNN Model Accuracy
    score = accuracy_score(y_test, knn_pred) * 100
    print(f"Accuracy using Knn Tree: {round(score, 1)}%")

    # KNN Confusion Matrix 
    results = confusion_matrix(y_test, knn_pred) 
    print('Confusion Matrix :')
    print(results) 
    print('Report for KNN Model: ')
    print(classification_report(y_test, knn_pred))




###################################################

# PREDICTING WHICH SONGS I WOULD LIKE
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials #To access authorised Spotify data


client_id = "4b15fcc288d3420abde032fde2e986ef" #Myat's Spotify ID 
client_secret = "6858e0e37841416684238be3c8d09e35" #Myat's Spotify Secret Key
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) #spotify object to access API

from spotify_API import playlist_trackID
from spotify_API import get_audio_features
from spotify_API import merge_dataframes
from spotify_API import main

first_dataframe_spotify = playlist_trackID('spotify','37i9dQZF1DXdwmD5Q7Gxah')
second_dataframe_spotify = get_audio_features(first_dataframe_spotify)
final_dataframe_spotify = merge_dataframes(first_dataframe_spotify,second_dataframe_spotify)

RF_CLF = RandomForestClassifier()
RF_CLF.fit(X_train, y_train)
pred = RF_CLF.predict(final_dataframe_spotify[variables])

def recc_songs():
    likedSongs = 0
    i = 0   
    for prediction in pred:
        if(prediction == 1):
            print ("Song: " + final_dataframe_spotify["track_name"][i] + ", By: "+ final_dataframe_spotify["artist_name"][i])
            #sp.user_playlist_add_tracks("1287242681", "7eIX1zvtpZR3M3rYFVA7DF", [test['id'][i]])
            likedSongs= likedSongs + 1
        i = i +1


def main():
    good = good_playlist('Carmen_Spotify.csv') #use print to see the dataframe
    bad = bad_playlist('BadList_Spotify.csv') #use print to see the dataframe
    # print(heatMap(good))
    # print(heatMap(bad))
    # good_hist('Carmen_Spotify.csv')
    # bad_hist('BadList_Spotify.csv')
    df_1 = 'Carmen_Spotify.csv'
    df_2 = 'BadList_Spotify.csv'
    one_playlist(df_1, df_2) #use print to see the dataframe
    remove_col(df_1, df_2)
    variable_type(df_1, df_2) #use print to see the dataframe

    print(decision_tree()) #use print to see accuracy results
    print(random_forest())
    print(knn_model())
    print(recc_songs())



if __name__ == '_main_':
    main()