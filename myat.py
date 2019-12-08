import spotipy
from spotipy.oauth2 import SpotifyClientCredentials #To access authorised Spotify data
import pandas as pd

client_id = "4b15fcc288d3420abde032fde2e986ef" #Myat's Spotify ID 
client_secret = "6858e0e37841416684238be3c8d09e35" #Myat's Spotify Secret Key
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) #spotify object to access API

def playlist_trackID(user_ID,playlist_ID):
    """
    This function takes a playlist ID, and the user's ID and returns a track id, popularity, artist name, and track name for each 
    track in the playlist. 
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
    This function gets all the audio features of a song. 
    """
    afeatures = []
    number = len(dataframe_name)
    for i in range(0,len(dataframe_name['track_id']),number):
        batch = dataframe_name['track_id'][i:i+number]
        audio_features = sp.audio_features(batch)
        for i, t in enumerate(audio_features):
            afeatures.append(t)
    return pd.DataFrame.from_dict(afeatures,orient='columns')

def merge_dataframes(dataframe1,dataframe2):
    """
    This function serves to merge the two dataframes created. It will first drop the unnecessary columns of the 
    dataframe that has the audio features, and rename the id so it can succesfully merge the two dataframes. 
    It is vital that the arguments in the function are correct.
    """
    drop_columns = ['analysis_url','track_href','type','uri']
    dataframe2.drop(drop_columns,axis=1,inplace=True)
    dataframe2.rename(columns={'id': 'track_id'}, inplace=True)
    return pd.merge(dataframe1,dataframe2,on='track_id',how='inner')

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

if __name__ == '__main__':
    main()