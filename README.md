# Final_Python_Project
This is a group repository for our final python project.

Instructions for use:

1) You will need to sign up for your own API keys at https://developer.spotify.com/

2) The list of packages and libraries that are needed is listed here: 
- Spotipy
    import: SpotifyClientCredentials, oauth2, util
- Pandas
- matplotlib
    import: pyplot
- seaborn
- sklearn 
    import: train_test_split, DecisionTreeClassifier, RandomForestClassifier, KNeighborClassifier, accuracy_score, confusion_matrix, classification_report

3) In place of "SECRET ID," you will need to input your own API keys (Client ID and Client Secret) provided by Spotify Developer.

4) For "USER_ID" and "PLAYLIST_ID," you will need to input your own Spotify username as well as your chosen playlist ID. This can be found in the URL. 

5) After running the three prediction models, choose the model with the highest accuracy to perform the actual prediction with. In our case, the Random Forest Tree was the highest accuracy. 

6) In the function recc_songs, you will need to input the playlist ID and User ID you would like to pick songs from to create your customized list. 

😎 For the "RF_CLF = RandomForestClassifier()" you will need to replace the RandomForestClassifier() with the model that had the highest accuracy after running.

OR

You can also find the instructions on how to run our code on our Jupyter Notebook - "Spotify_Prediction".

You can also learn about our journey to completing this project at our website: https://sites.google.com/view/team-soy-sauce/home