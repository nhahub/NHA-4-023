import streamlit as st
import streamlit.components.v1 as components
import pickle
import requests
from rapidfuzz import process, fuzz
from collections import defaultdict
from surprise import dump
# =================================================================
# variables
# =================================================================

movies =  pickle.load(open("movies_features.pkl" , "rb"))
cosine_sim =  pickle.load(open("movies_cosine_sim.pkl" , "rb"))
testset =  pickle.load(open("test_prediction_users.pkl" , "rb"))
# algo =  pickle.load(open("algo_prediction_users.pkl" , "rb"))

predictions= dump.load("prediction_users.pkl" )
userId_list =  pickle.load(open("user_id_list.pkl", "rb"))
movies_id_title = movies[["movieId", "title"]]

populary_movies =  pickle.load(open("populary_movies.pkl" , "rb"))

movies_list = movies["title"].values


# =================================================================
# defined functions
# =================================================================
# fetch image as full path from The movie db API
# ================================================
def fetch_info(movie_id):
    
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiZGU2MjlmMzNmOGM2ZTRlZWE0ZDdlY2FjMDhkYjE3ZCIsIm5iZiI6MTc3OTA0NTEyOC4xNCwic3ViIjoiNmEwYTEzMDgxNzU2YTRmOWFlZGIyZTQyIiwic2NvcGVzIjpbImFwaV9yZWFkIl0sInZlcnNpb24iOjF9.fWFsXyCMSKwTN38zPszVONozLb6qw5HJQGKvsfAC0gg"
    }

    response = requests.get(url, headers=headers)

    # print(response.text)
    data = response.json()
    moive_genres = [ i["name"] for i in data["genres"]]
    moive_overview = data["overview"]
    moive_release_date = data["release_date"]
    moive_run_time = f"{data["runtime"]} mins"
    moive_vote_ave = f"{data["vote_average"]} / 10"
    moive_vote_count = f"{data["vote_count"]} votes"

    movie_info_list = [moive_genres , moive_overview , moive_release_date , moive_run_time , moive_vote_ave , moive_vote_count] 

    return movie_info_list
# =================================================================
# fetch image as full path from The movie db API
# ================================================
def fetch_poster(movie_id):
    
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiZGU2MjlmMzNmOGM2ZTRlZWE0ZDdlY2FjMDhkYjE3ZCIsIm5iZiI6MTc3OTA0NTEyOC4xNCwic3ViIjoiNmEwYTEzMDgxNzU2YTRmOWFlZGIyZTQyIiwic2NvcGVzIjpbImFwaV9yZWFkIl0sInZlcnNpb24iOjF9.fWFsXyCMSKwTN38zPszVONozLb6qw5HJQGKvsfAC0gg"
    }

    response = requests.get(url, headers=headers)

    # print(response.text)
    data = response.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/"+poster_path
    return full_path

# =========================================================================
# function return 5 recommended movies with content based filtering method
# =========================================================================

def movie_name_finder(title):
    all_titles = movies['title'].tolist()
    closest_match = process.extract(title, all_titles,scorer=fuzz.WRatio, limit=2)
    return closest_match[0][0]

# =========================================================================
def get_content_based_recommendations(movie_title, cosine_sim_matrix, df, top_n=5):
    idx = df.index[df['title'] == movie_name_finder(movie_title)].tolist()
    if not idx:
        return "Movie not found!"
    idx = idx[0]

    sim_scores = list(enumerate(cosine_sim_matrix[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n+1]
    movie_indices = list(i[0] for i in sim_scores)

    return movie_indices

# =================================================================
def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# =================================================================

def get_collaborative_filtering_recommendations(user_id, top_n):
    predictions = predictions
    my_pre = get_top_n(predictions, top_n)    

    my_list = []
    for i in my_pre[user_id]:
        my_list.append(i[0]) 
       
    return my_list

# =================================================================
def get_title_from_movieid(movieId):
    
    movie_title = movies_id_title[movies_id_title["movieId"] == movieId]["title"].values[0]

    return movie_title


def get_movieid_from_title(movie_title):
    
    movieId = movies_id_title[movies_id_title["title"] == movie_title]["title"].values[0]

    return movieId
# =================================================================


# Hybrid Approach
def get_hybrid_recommendations(user_id, movie_name, top_n = 10):

    recommend_movie = []
    recommend_poster = []

    content_based_recommendations = get_content_based_recommendations(movie_name, cosine_sim, movies , top_n)
    collaborative_filtering_recommendations = get_collaborative_filtering_recommendations(user_id, top_n)
    hybrid_recommendations = list(set(content_based_recommendations + collaborative_filtering_recommendations))
    
    for i in hybrid_recommendations[:top_n]:

        recommend_movie.append(movies['title'].iloc[i[0]])

        movies_id = movies["tmdbId"].iloc[i[0]]

        recommend_poster.append(fetch_poster(movies_id))
    
    return recommend_movie , recommend_poster


# =================================================================

def get_main_movie_info(movie_name):

    idx = movies.index[movies['title'] == movie_name_finder(movie_name)].tolist()
    if not idx:
        return "Movie not found!"
    idx = idx[0]
    selected_movie_info = fetch_info(movies["tmdbId"].iloc[idx])
    moive_poster = fetch_poster(movies["tmdbId"].iloc[idx])

    return selected_movie_info , moive_poster
    









# =================================================================
# streamlit web app interface
# =================================================================
st.header("Movie Recommender System")

# imageCarouselComponent = components.declare_component("image-carousel-component")



st.header("Top 10 Popular Movies")


# ============================================
# popularity based recommendation
# ============================================

imageCarouselComponent = components.declare_component("image-carousel-component", path="frontend/public")

imageUrls = [    fetch_poster(moive_id) for moive_id in populary_movies[ "tmdbId" ].to_list()[0:10]  ]

# imageUrls = [
#     fetch_poster(1632),
#     fetch_poster(299536),
#     fetch_poster(17455),
#     fetch_poster(2830),
#     fetch_poster(429422),
#     fetch_poster(9722),
#     fetch_poster(13972),
#     fetch_poster(240),
#     fetch_poster(155),
#     fetch_poster(598),
#     fetch_poster(914),
#     fetch_poster(255709),
#     fetch_poster(572154)
   
#     ]

imageCarouselComponent(imageUrls=imageUrls, height = 250)
# imageCarouselComponent(imageUrls=imageUrls, height=200)


# ============================================
# recommendation button and 5 output
# ============================================

select_value_userId = st.selectbox("Select user from dropdown", userId_list)
select_value_moive = st.selectbox("Select movie from dropdown", movies_list)

if st.button("Show Recommendations"):
    selected_movie, movie_poster = get_main_movie_info(select_value_moive)

    moives_recom , movie_poster = get_hybrid_recommendations(select_value_userId, select_value_moive, top_n = 10)

    st.subheader(selected_movie)
    col1,col2 = st.columns(2)
    with col1:
        st.image(movie_poster)
    with col2:
        # for i in range(0,len(selected_movie)):
        #     st.text(selected_movie[i])
        #[moive_genres , moive_overview , moive_release_date , moive_run_time , moive_vote_ave , moive_vote_count] 
        st.text(f"Genres: {", ".join(selected_movie[0])}")  
        st.text(f"Overview: {selected_movie[1]}")  
        st.text(f"Release Date: {selected_movie[2]}")  
        st.text(f"Run Time: {selected_movie[3]}")  
        st.text(f"Average Voting Rate: {selected_movie[4]}")  
        st.text(f"Total Votes: {selected_movie[5]}")  

    cols_1 = st.columns(5)
    for i in range(1 ,5):
        with cols_1[i]:
            st.image(movie_poster[i])
            st.text(moives_recom[i])            

    # col1,col2,col3,col4,col5 = st.columns(5)

    # with col1:
    #     st.image(movie_poster[1])
    #     st.text(moives_recom[1])
    # with col2:
    #     st.image(movie_poster[2])
    #     st.text(moives_recom[2])
    # with col3:
    #     st.image(movie_poster[3])
    #     st.text(moives_recom[3])
    # with col4:
    #     st.image(movie_poster[4])
    #     st.text(moives_recom[4])
    # with col5:
    #     st.image(movie_poster[5])
    #     st.text(moives_recom[5])