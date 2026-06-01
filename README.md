# NHA-4-23
# 🎬 CineMatch - Hybrid Movie Recommendation System

CineMatch is a movie recommendation system that combines multiple recommendation techniques to provide personalized movie suggestions.

The project implements:

* Content-Based Filtering
* Collaborative Filtering (SVD Matrix Factorization)
* Popularity-Based Recommendation
* Hybrid Recommendation Engine
* Streamlit Web Application

The application allows users to select a movie they like and their user ID, then generates recommendations based on both their historical preferences and movie similarity.

---

# 📂 Project Structure

```text
├── Recommendation_Struc_V6_Hamdy.ipynb   # Model development notebook
├── app_3.py                              # Streamlit application
├── movies_features.pkl                   # Processed movie features
├── movies_cosine_sim.pkl                 # Cosine similarity matrix
├── top_n_all.pkl                         # Precomputed CF recommendations
├── user_id_list.pkl                      # Available user IDs
├── populary_movies.pkl                   # Popular movies dataset
├── requirements.txt
└── README.md
```

---

# 🚀 Recommendation Approaches

## 1. Content-Based Filtering

Content-Based Filtering recommends movies that are similar to a movie the user already likes.

### Features Used

The movie feature vector is built using:

* Genres (One-Hot Encoded)
* Average Rating
* Number of Ratings
* Bayesian Average Rating
* Release Year

### Process

1. Extract movie genres.
2. Apply MultiLabelBinarizer to encode genres.
3. Merge movie metadata.
4. Create a feature matrix.
5. Calculate cosine similarity between movies.
6. Recommend the most similar movies.

### Example

If a user likes:

```text
Toy Story
```

The system recommends movies with similar genres and characteristics.

---

## 2. Collaborative Filtering

Collaborative Filtering recommends movies based on the behavior of similar users.

### Algorithm

The project uses:

```text
SVD (Singular Value Decomposition)
```

from the Surprise library.

### Process

1. Build a User-Movie Rating Matrix.
2. Train an SVD model.
3. Learn latent factors for users and movies.
4. Predict unseen movie ratings.
5. Generate Top-N recommendations for each user.

### Evaluation Metrics

The model is evaluated using:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)

---

## 3. Popularity-Based Recommendation

Popularity recommendations are used when user history is unavailable.

### Ranking Criteria

Movies are ranked using:

* Average Rating
* Number of Ratings
* Bayesian Average Rating

### Why Bayesian Average?

A movie with only 2 ratings and a perfect score should not rank higher than a movie with thousands of ratings.

The Bayesian Average balances:

* Rating quality
* Rating quantity

This creates a more reliable popularity score.

---

# 🔥 Cold Start Problem

One of the biggest challenges in recommendation systems is the Cold Start Problem.

---

## New User Cold Start

### Problem

A new user has not rated any movies.

### Solution

Use Popularity-Based Recommendations.

The application displays trending and highly rated movies using the Bayesian ranking method.

This ensures new users immediately receive useful recommendations.

---

## New Movie Cold Start

### Problem

A newly added movie has no ratings.

Collaborative Filtering cannot recommend it because no user interactions exist.

### Solution

Use Content-Based Filtering.

Since content-based recommendations rely on movie attributes rather than user ratings, the system can recommend new movies using:

* Genre
* Release Year
* Metadata Features

As soon as the movie is added to the catalog, it can participate in recommendations.

---

# 🤝 Hybrid Recommendation Strategy

The final recommendation engine combines:

```text
Content-Based Filtering
+
Collaborative Filtering
```

### Workflow

1. User selects a movie they like.
2. System generates Content-Based recommendations.
3. System generates Collaborative Filtering recommendations.
4. Results are merged and deduplicated.
5. Top recommendations are displayed.

### Benefits

Content-Based Filtering captures:

* Movie similarity

Collaborative Filtering captures:

* User preference patterns

Combining both methods improves recommendation quality and diversity.

---

# 🎯 Recommendation Based on User History

The system leverages historical ratings through Collaborative Filtering.

### How it Works

For a given user:

1. Retrieve past ratings.
2. Use the trained SVD model to estimate ratings for unseen movies.
3. Rank predicted ratings.
4. Return Top-N movies.

This allows the system to learn:

* Preferred genres
* Rating behavior
* Similar users

and recommend movies the user is likely to enjoy.

---

# 📊 Data Preparation Pipeline

## Step 1: Load Data

Datasets used:

```text
movies.csv
ratings.csv
links.csv
```

---

## Step 2: Exploratory Data Analysis

Performed analysis on:

* Number of users
* Number of movies
* Number of ratings
* Rating distribution
* Most rated movies
* Genre popularity

---

## Step 3: Feature Engineering

Created additional features:

### Movie Features

* Release Year
* Average Rating
* Number of Ratings
* Bayesian Average Rating

### Genre Features

Genres were transformed using:

```python
MultiLabelBinarizer()
```

to create machine-readable vectors.

---

## Step 4: Similarity Matrix

Cosine similarity was computed on the movie feature matrix:

```python
cosine_similarity(features_matrix)
```

This matrix is used by the Content-Based recommender.

---

## Step 5: Train SVD Model

Using Surprise:

```python
from surprise import SVD
```

Hyperparameter tuning was performed using Grid Search.

The best model was retrained on the full dataset.

---

## Step 6: Export Artifacts

Generated and saved:

```text
movies_features.pkl
movies_cosine_sim.pkl
top_n_all.pkl
user_id_list.pkl
populary_movies.pkl
```

These files are loaded directly by the Streamlit application.

---

# 🌐 Streamlit Application

The application provides:

### Trending Movies

Displays popular movies for cold-start users.

### Movie Search

Users can select a movie they like.

### Personalized Recommendations

Uses the hybrid recommendation engine.

### Movie Metadata

Fetched from TMDB API:

* Poster
* Genres
* Overview
* Runtime
* Release Date
* Ratings

---

# ⚙️ Create Virtual Environment

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-surprise rapidfuzz requests streamlit
```

---

# ▶️ Run the Application

```bash
streamlit run app_3.py
```

---

# Future Improvements

* User authentication
* Real-time feedback collection
* Deep Learning recommenders
* Transformer-based embeddings
* Explainable recommendations
* Online model retraining

---

# Author

**Mohamed Hamdy**

Hybrid Movie Recommendation System using Content-Based Filtering, Collaborative Filtering (SVD), and Popularity-Based Recommendation deployed with Streamlit.

