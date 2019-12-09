from flask import Flask, render_template, request
from vicky import recc_songs

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/spotify_playlist/", methods = ["GET", "POST"])
def features():
    if request.method == "POST":
        feature1 = str(request.form["playlist_info"])
        dataframe1, dataframe2 = merge_dataframes(dataframe1, dataframe2)
        value = ""
        if dataframe1:
                drop_columns = ['analysis_url','track_href','type','uri']
                dataframe2.drop(drop_columns,axis=1,inplace=True)
                dataframe2.rename(columns={'id': 'track_id'}, inplace=True)
                return render_template("response.html", )
        
        else:
            return render_template("next.html", error = True)
    return render_template("next.html", error = None)
    