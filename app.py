import streamlit as st
import streamlit.components.v1 as components
import pickle
import requests
from rapidfuzz import process, fuzz
import pandas as pd

# =================================================================
# Page Config
# =================================================================
st.set_page_config(
    page_title="CineMatch · Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =================================================================
# Custom CSS — cinematic dark theme
# =================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    background-color: #0a0a0f;
    color: #e8e4dc;
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 2px; }

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #0a0a0f 0%, #1a0a2e 50%, #0d1117 100%);
    border-bottom: 1px solid #f5c518;
    padding: 2rem 2rem 1.5rem;
    margin: -1rem -1rem 2rem;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    letter-spacing: 6px;
    color: #f5c518;
    margin: 0;
    line-height: 1;
}
.hero-sub {
    color: #888;
    font-size: 0.85rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Section headers */
.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    letter-spacing: 3px;
    color: #f5c518;
    border-left: 3px solid #f5c518;
    padding-left: 12px;
    margin: 2rem 0 1rem;
}

/* Movie card */
.movie-card {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 0;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
    cursor: pointer;
}
.movie-card:hover { transform: translateY(-4px); border-color: #f5c518; }
.movie-card img { width: 100%; display: block; }
.movie-card-title {
    font-size: 0.78rem;
    padding: 8px 8px 10px;
    color: #c8c4bc;
    line-height: 1.3;
    min-height: 42px;
}

/* Main movie info */
.info-box {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 1.5rem;
}
.info-label {
    color: #f5c518;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 2px;
}
.info-value { color: #e8e4dc; font-size: 0.9rem; margin-bottom: 1rem; }
.info-overview { color: #a8a4a0; font-size: 0.88rem; line-height: 1.6; }
.genre-pill {
    display: inline-block;
    background: #1e1e2e;
    border: 1px solid #f5c518;
    color: #f5c518;
    font-size: 0.7rem;
    letter-spacing: 1px;
    padding: 2px 10px;
    border-radius: 20px;
    margin: 2px 3px 2px 0;
}
.rating-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1e1e2e;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}
.star { color: #f5c518; }

/* Selectbox + button */
div[data-baseweb="select"] > div {
    background-color: #111118 !important;
    border-color: #1e1e2e !important;
    color: #e8e4dc !important;
}
.stButton > button {
    background: #f5c518;
    color: #0a0a0f;
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 3px;
    font-size: 1rem;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 2rem;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover { background: #ffd740; }

/* Error / warning */
.error-box {
    background: #1a0a0a;
    border: 1px solid #c0392b;
    border-radius: 6px;
    padding: 1rem;
    color: #e74c3c;
    font-size: 0.85rem;
}

/* Divider */
.gold-divider {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# =================================================================
# Load Artifacts
# =================================================================

@st.cache_resource
def load_top_n():
    with open("top_n_all.pkl", "rb") as f:
        return pickle.load(f)

# @st.cache_data
# def load_movies():
#     return pd.read_pickle("movies_features.pkl")

@st.cache_resource
def load_movies():
    with open("movies_features.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_cosine_sim():
    with open("movies_cosine_sim.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_user_id_list():
    with open("user_id_list.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_popular_movies():
    with open("populary_movies.pkl", "rb") as f:
        return pickle.load(f)

top_n_all       = load_top_n()
movies          = load_movies()
cosine_sim      = load_cosine_sim()
userId_list     = load_user_id_list()
popular_movies  = load_popular_movies()
movies_id_title = movies[["movieId", "title"]]
movies_list     = movies["title"].values

# =================================================================
# API — single call returning both poster + info
# =================================================================

FALLBACK_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiZGU2MjlmMzNmOGM2ZTRlZWE0ZDdlY2FjMDhkYjE3ZCIsIm5iZiI6MTc3OTA0NTEyOC4xNCwic3ViIjoiNmEwYTEzMDgxNzU2YTRmOWFlZGIyZTQyIiwic2NvcGVzIjpbImFwaV9yZWFkIl0sInZlcnNpb24iOjF9.fWFsXyCMSKwTN38zPszVONozLb6qw5HJQGKvsfAC0gg"
try:
    TMDB_TOKEN = st.secrets["TMDB_TOKEN"]
except Exception:
    TMDB_TOKEN = FALLBACK_TOKEN
TMDB_HEADERS = {"accept": "application/json", "Authorization": f"Bearer {TMDB_TOKEN}"}
POSTER_BASE  = "https://image.tmdb.org/t/p/w500"
FALLBACK_IMG = "https://via.placeholder.com/500x750/111118/f5c518?text=No+Poster"

@st.cache_data(show_spinner=False)
def fetch_movie_data(tmdb_id):
    """Single TMDB call returning poster URL + metadata dict."""
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}?language=en-US",
            headers=TMDB_HEADERS, timeout=5
        )
        r.raise_for_status()
        d = r.json()
        poster = f"{POSTER_BASE}{d['poster_path']}" if d.get("poster_path") else FALLBACK_IMG
        return {
            "poster":       poster,
            "genres":       [g["name"] for g in d.get("genres", [])],
            "overview":     d.get("overview", "No overview available."),
            "release_date": d.get("release_date", "—"),
            "runtime":      f"{d.get('runtime', '?')} min",
            "vote_avg":     round(d.get("vote_average", 0), 1),
            "vote_count":   f"{d.get('vote_count', 0):,}",
        }
    except Exception:
        return {"poster": FALLBACK_IMG, "genres": [], "overview": "", 
                "release_date": "—", "runtime": "—", "vote_avg": 0, "vote_count": "—"}

# =================================================================
# Recommendation Logic
# =================================================================

def movie_name_finder(title):
    match = process.extract(title, movies["title"].tolist(), scorer=fuzz.WRatio, limit=1)
    return match[0][0] if match else title

def get_content_based_recommendations(movie_title, top_n=10):
    matched = movie_name_finder(movie_title)
    idx_list = movies.index[movies["title"] == matched].tolist()
    if not idx_list:
        return []
    idx = idx_list[0]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:top_n+1]
    return movies["movieId"].iloc[[i for i, _ in sim_scores]].tolist()

def get_collaborative_filtering_recommendations(user_id, n=10):
    if user_id not in top_n_all:
        return []
    return [iid for iid, _ in top_n_all[user_id][:n]]

def get_hybrid_recommendations(user_id, movie_name, top_n=10):
    cb   = get_content_based_recommendations(movie_name, top_n)
    cf   = get_collaborative_filtering_recommendations(user_id, top_n)
    # Interleave CB and CF to balance both signals, then deduplicate
    merged, seen = [], set()
    for a, b in zip(cb, cf):
        for mid in (a, b):
            if mid not in seen:
                merged.append(mid)
                seen.add(mid)
    for mid in cb + cf:
        if mid not in seen:
            merged.append(mid)
            seen.add(mid)
    return merged[:top_n]

def get_tmdb_id(movie_id):
    row = movies[movies["movieId"] == movie_id]["tmdbId"].values
    return int(row[0]) if len(row) else None

def get_title(movie_id):
    row = movies_id_title[movies_id_title["movieId"] == movie_id]["title"].values
    return row[0] if len(row) else "Unknown"

# =================================================================
# UI
# =================================================================

# Hero banner
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">CineMatch</p>
    <p class="hero-sub">Hybrid Movie Recommendation Engine</p>
</div>
""", unsafe_allow_html=True)

# =================================================================
# ── Popular Movies ────────────────────────────────────────────────
# =================================================================
st.markdown('<p class="section-title">🔥 Trending Now</p>', unsafe_allow_html=True)

try:
    imageCarouselComponent = components.declare_component(
        "image-carousel-component", path="frontend/public"
    )
    with st.spinner("Loading popular movies…"):
        # M_filter = (popular_movies['Comedy'] == 1 ) & (popular_movies['Animation'] == 1 )
        popular_tmdb_ids = popular_movies["tmdbId"].tolist()[:10]
        imageUrls = [fetch_movie_data(tid)["poster"] for tid in popular_tmdb_ids]
    imageCarouselComponent(imageUrls=imageUrls, height=250)
except Exception:
    st.info("Carousel component not available — showing grid instead.")
    pop_cols = st.columns(5)
    for i, col in enumerate(pop_cols):
        with col:
            tid = popular_movies["tmdbId"].tolist()[i]
            d   = fetch_movie_data(tid)
            st.image(d["poster"], use_container_width=True)

st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

# =================================================================
# ── Search Controls ───────────────────────────────────────────────
# =================================================================
st.markdown('<p class="section-title">🎯 Get Recommendations</p>', unsafe_allow_html=True)

col_a, col_b, col_c, col_d = st.columns([2, 2, 1, 1])
with col_a:
    select_value_userId = st.selectbox("👤 Your User ID", userId_list)
with col_b:
    select_value_movie = st.selectbox("🎬 Pick a Movie You Like", movies_list)
with col_c:
    num_recs = st.selectbox("🎞️ How Many?", [5, 10, 20], index=1)
with col_d:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Find Matches →")

# ── Results ───────────────────────────────────────────────────────
if run:
    with st.spinner("Analysing your taste…"):
        matched_title = movie_name_finder(select_value_movie)
        idx_list = movies.index[movies["title"] == matched_title].tolist()

        if not idx_list:
            st.markdown('<div class="error-box">⚠️ Movie not found in database.</div>', 
                        unsafe_allow_html=True)
            st.stop()

        seed_tmdb = get_tmdb_id(movies["movieId"].iloc[idx_list[0]])
        seed_data = fetch_movie_data(seed_tmdb)

        recs = get_hybrid_recommendations(select_value_userId, select_value_movie, top_n=num_recs)

    # ── Seed movie card ──────────────────────────────────────────
    st.markdown('<p class="section-title">📽️ Selected Movie</p>', unsafe_allow_html=True)
    left, right = st.columns([1, 3])

    with left:
        st.image(seed_data["poster"], use_container_width=True)

    with right:
        st.markdown(f"### {matched_title}")
        genres_html = "".join(f'<span class="genre-pill">{g}</span>' for g in seed_data["genres"])
        st.markdown(f'<div style="margin-bottom:1rem">{genres_html}</div>', unsafe_allow_html=True)

        st.markdown(
            f'<span class="rating-badge"><span class="star">★</span> '
            f'{seed_data["vote_avg"]} &nbsp;·&nbsp; {seed_data["vote_count"]} votes</span>',
            unsafe_allow_html=True
        )
        st.markdown(f"""
<div class="info-box" style="margin-top:1rem">
    <div class="info-label">Overview</div>
    <div class="info-overview">{seed_data["overview"]}</div>
    <br>
    <div style="display:flex; gap:2rem; flex-wrap:wrap">
        <div>
            <div class="info-label">Release Date</div>
            <div class="info-value">{seed_data["release_date"]}</div>
        </div>
        <div>
            <div class="info-label">Runtime</div>
            <div class="info-value">{seed_data["runtime"]}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)

    # ── Recommendations grid ─────────────────────────────────────
    st.markdown('<p class="section-title">✨ Recommended For You</p>', unsafe_allow_html=True)

    if not recs:
        st.markdown('<div class="error-box">⚠️ No recommendations found for this user/movie combination.</div>',
                    unsafe_allow_html=True)
    else:
        with st.spinner("Fetching movie details…"):
            rec_data = []
            for mid in recs:
                tmdb_id = get_tmdb_id(mid)
                title   = get_title(mid)
                info    = fetch_movie_data(tmdb_id) if tmdb_id else {"poster": FALLBACK_IMG}
                rec_data.append({"title": title, "poster": info["poster"],
                                  "rating": info.get("vote_avg", ""), 
                                  "genres": info.get("genres", [])})

        # Dynamic grid — always 5 columns per row
        row_size = 5
        for row_start in range(0, len(rec_data), row_size):
            row_items = rec_data[row_start:row_start + row_size]
            cols = st.columns(row_size)
            for col, item in zip(cols, row_items):
                with col:
                    st.image(item["poster"], use_container_width=True)
                    rating_str = f" · ★ {item['rating']}" if item["rating"] else ""
                    st.caption(f"{item['title']}{rating_str}")
