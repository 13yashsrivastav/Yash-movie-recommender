import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide")

# ---------------- BACKGROUND STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg,#141E30,#243B55);
    color:white;
}

.big-title {
    font-size:45px;
    font-weight:bold;
    text-align:center;
    padding:20px;
}

.movie-card {
    background-color:#1c1c1c;
    padding:10px;
    border-radius:12px;
    text-align:center;
    box-shadow:0px 0px 10px rgba(0,0,0,0.6);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🎬 Yash Movie Recommender System</div>', unsafe_allow_html=True)
st.write("Pick a movie — discover similar ones 🍿")

# ---------------- OMDB POSTER ----------------
API_KEY = "980868a1"

def fetch_poster(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
    data = requests.get(url).json()
    return data.get("Poster")

# ---------------- LOAD DATA ----------------
movies = pd.read_csv("tmdb_5000_movies.csv")
movies = movies[['title','genres','keywords','overview']]

def convert(text):
    L=[]
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies['tags'] = movies['genres'] + movies['keywords'] + movies['overview']
movies['tags'] = movies['tags'].apply(lambda x:" ".join(x))

# ---------------- VECTOR ----------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# ---------------- RECOMMEND ----------------
def recommend(movie):
    idx = movies[movies['title'] == movie].index[0]
    distances = list(enumerate(similarity[idx]))
    movies_list = sorted(distances, reverse=True, key=lambda x:x[1])[1:6]
    return [movies.iloc[i[0]].title for i in movies_list]

# ---------------- UI ----------------
selected_movie = st.selectbox("Select a Movie", movies['title'].values)

if st.button("Recommend 🎯"):
    names = recommend(selected_movie)

    cols = st.columns(5)

    for i in range(5):
        poster = fetch_poster(names[i])
        with cols[i]:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)

            if poster and poster != "N/A":
                st.image(poster)
            else:
                st.write("No Image")

            st.write(names[i])
            st.markdown('</div>', unsafe_allow_html=True)
