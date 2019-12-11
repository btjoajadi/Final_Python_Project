##################################################################################
# Packages
##################################################################################

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials #To access authorised Spotify data
import pandas as pd

##################################################################################
# API Keys
##################################################################################

SECRET_ID = str(input('Please enter your Client ID: '))
SECRET_ID2 = str(input('Please enter your Client Secret ID: '))

client_id = SECRET_ID
client_secret = SECRET_ID2
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) #spotify object to access API


def playlist_trackID(user_ID,playlist_ID):
    """
    This function will take a playlist ID and a user's ID and return a Pandas DataFrame of the song's features within that playlist 
    """
    track_id = []
    popularity = []
    artist_name = []
    track_name = []
    playlist = sp.user_playlist_tracks(user_ID,playlist_ID)
    for song in playlist['items']:
        track = song['track']
        track_id.append(track['id'])
        popularity.append(track['popularity'])
        track_name.append(track['name'])
        artist_name.append(track['artists'][0]['name'])
    return pd.DataFrame({'artist_name':artist_name,'track_name':track_name,'track_id':track_id,'popularity':popularity})


def get_audio_features(dataframe_name):
    """
    Pulling all of the audio features of a song into a list and then into a dataframe
    """
    afeatures = []
    number = len(dataframe_name)
    for i in range(0,len(dataframe_name['track_id']),number):
        batch = dataframe_name['track_id'][i:i+number]
        audio_features = sp.audio_features(batch)
        for i, t in enumerate(audio_features):
            afeatures.append(t)
    return pd.DataFrame.from_dict(afeatures,orient='columns')


def get_audio_features(dataframe_name):
    """
    Pulling all of the audio features of a song into a list and then into a dataframe
    """
    afeatures = []
    number = len(dataframe_name)
    for i in range(0,len(dataframe_name['track_id']),number):
        batch = dataframe_name['track_id'][i:i+number]
        audio_features = sp.audio_features(batch)
        for i, t in enumerate(audio_features):
            afeatures.append(t)
    return pd.DataFrame.from_dict(afeatures,orient='columns')



##################################################################################
# Packages
##################################################################################

#packages & libraries
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 

##################################################################################
# Data Pre-Processing
##################################################################################


def good_playlist(good_playlist_csv):
    '''
    Making a new column called "TARGET" that classifies the good and bad playlists 
    good : playlist of songs that the individual likes
    '''
    good = pd.read_csv(good_playlist_csv)
    target = 1
    good['target'] = target
    return good


def bad_playlist(bad_playlist_csv):
    '''
    Making a new column called "TARGET" that classifies the good and bad playlists 
    bad: playlist of songs that the individual dislikes 
    '''
    bad = pd.read_csv(bad_playlist_csv)
    target = 0
    bad['target'] = target
    return bad


def correlation_map(df):
    '''
    correlation map that checks the relationship between variables
    '''
    corr = df.corr()   
    fig, ax = plt.subplots(figsize=(10, 10))     
    colormap = sns.diverging_palette(220, 10, as_cmap=True)   
    sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")   
    plt.xticks(range(len(corr.columns)),   
    corr.columns);     
    plt.yticks(range(len(corr.columns)), corr.columns)
    return plt.show()   


def good_hist(df):
    '''
    Histograms displaying the distribution of values in each characteristic of the good playlist
    '''
    good = good_playlist(df)
    good.hist(alpha=0.7, label='positive')
    plt.legend(loc='upper right')
    return plt.show()


def bad_hist(df):
    '''
    Histograms displaying the distribution of values in each characteristic of the bad playlist
    '''
    bad = bad_playlist(df)
    bad.hist(alpha=0.7, label='negative')
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
    Deleting the unnecessary columns that simply label row IDs
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


def decision_tree(df_1, df_2):
    # Decision Tree Classification Model
    new_rr = remove_col(df_1, df_2)
    random_seed = 5 #set random seed for reproducible results 
    variables = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness","duration_ms"]
    X = new_rr[variables] #using the variables we would like to use 
    y = new_rr["target"] #target variable 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed) # 80% training and 20% test

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


def random_forest(df_1, df_2):
    # Random Forest Tree Model
    new_rr = remove_col(df_1, df_2)
    random_seed = 5 #set random seed for reproducible results 
    variables = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness","duration_ms"]
    X = new_rr[variables] #using the variables we would like to use 
    y = new_rr["target"] #target variable 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed) # 80% training and 20% test

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


def knn_model(df_1, df_2):
    #KNN Model
    new_rr = remove_col(df_1, df_2)
    random_seed = 5 #set random seed for reproducible results 
    variables = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness","duration_ms"]
    X = new_rr[variables] #using the variables we would like to use 
    y = new_rr["target"] #target variable 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed) # 80% training and 20% test

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

def merge_dataframes(dataframe1,dataframe2):
    """
    This function serves to merge the two dataframes created. It will first drop the unnecessary columns of the 
    dataframe that has the audio features, and rename the id so it can succesfully merge the two dataframes. 
    It is vital that the arguments in the function are correct.
    """
    drop_columns = ['analysis_url','track_href','type']
    dataframe2.drop(drop_columns,axis=1,inplace=True)
    dataframe2.rename(columns={'id': 'track_id'}, inplace=True)
    return pd.merge(dataframe1,dataframe2,on='track_id',how='inner')


##################################################################################
# Prediciting Which Songs You Would Like From a Playlist
##################################################################################


first_dataframe_spotify = playlist_trackID('spotify','37i9dQZF1DXdwmD5Q7Gxah')
second_dataframe_spotify = get_audio_features(first_dataframe_spotify)
final_dataframe_spotify = merge_dataframes(first_dataframe_spotify,second_dataframe_spotify)

def recc_songs(df_1, df_2, df_3):
    new_rr = remove_col(df_1, df_2)
    random_seed = 5 #set random seed for reproducible results 
    variables = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness","duration_ms"]
    X = new_rr[variables] #using the variables we would like to use 
    y = new_rr["target"] #target variable 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed) # 80% training and 20% test
    RF_CLF = RandomForestClassifier()
    RF_CLF.fit(X_train, y_train)
    pred = RF_CLF.predict(df_3[variables])
    likedSongs = 0
    i = 0   
    artist_name = []
    track_id = []
    track_name =[]
    track_uri = []
    for prediction in pred:
        if(prediction == 1):
            artist_name.append(df_3["artist_name"][i])
            track_id.append(df_3["track_id"][i])
            track_name.append(df_3["track_name"][i])
            track_uri.append(df_3['uri'][i])
            # print ("Song: " + final_dataframe_spotify["track_name"][i] + ", By: "+ final_dataframe_spotify["artist_name"][i])
            likedSongs= likedSongs + 1
        i = i +1
    return pd.DataFrame({'artist_name':artist_name,'track_name':track_name,'track_id':track_id,'track_uri':track_uri})



def main():
    first_dataframe_myat = playlist_trackID('lookitschibbles','37i9dQZF1EjeiPPNs5t2ax')
    second_dataframe_myat = get_audio_features(first_dataframe_myat)
    final_dataframe_myat = merge_dataframes(first_dataframe_myat,second_dataframe_myat)
    final_dataframe_myat.to_csv('Myat_Spotify.csv') #note, cannot override, so must delete previous 

    first_dataframe_vicky = playlist_trackID('rcsq0l8zod45cf261mfijayil','37i9dQZF1EjjOJ0ymAONtH')
    second_dataframe_vicky = get_audio_features(first_dataframe_vicky)
    final_dataframe_vicky = merge_dataframes(first_dataframe_vicky,second_dataframe_vicky)
    final_dataframe_vicky.to_csv('Vicky_Spotify.csv')

    first_dataframe_carmen = playlist_trackID('carmenngo97','37i9dQZF1EjhZxbxJKzY51')
    second_dataframe_carmen = get_audio_features(first_dataframe_carmen)
    final_dataframe_carmen = merge_dataframes(first_dataframe_carmen,second_dataframe_carmen)
    final_dataframe_carmen.to_csv('Carmen_Spotify.csv')

    first_dataframe_bad = playlist_trackID('northofnowhere','0fnnYX71GUvWlMDKGX40FS')
    second_dataframe_bad = get_audio_features(first_dataframe_bad)
    final_dataframe_bad = merge_dataframes(first_dataframe_bad,second_dataframe_bad)
    final_dataframe_bad.to_csv('BadList_Spotify.csv')

    good = good_playlist('Vicky_Spotify.csv') 
    bad = bad_playlist('BadList_Spotify.csv') 
    correlation_map(good)
    correlation_map(bad)
    good_hist('Vicky_Spotify.csv')
    bad_hist('BadList_Spotify.csv')
    df_1 = 'Vicky_Spotify.csv'
    df_2 = 'BadList_Spotify.csv'
    one_playlist(df_1, df_2) #use print to see the dataframe
    remove_col(df_1, df_2)
    variable_type(df_1, df_2) #use print to see the dataframe

    decision_tree(df_1, df_2) #use print to see accuracy results
    random_forest(df_1, df_2)
    knn_model(df_1, df_2)

    reccy=recc_songs(df_1,df_2,final_dataframe_spotify)
    print(reccy)

if __name__ == '__main__':
    main()