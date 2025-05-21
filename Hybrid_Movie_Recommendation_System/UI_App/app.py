import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler

@st.cache_data
def load_data():
    movies = pd.read_csv("Raw_DataSets/movies.csv")
    ratings = pd.read_csv("Raw_DataSets/ratings.csv")
    return movies, ratings

@st.cache_data
def preprocess_movies(movies):
    movies['genres'] = movies['genres'].fillna('')
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    return movies

@st.cache_data
def build_tfidf_matrix(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    return tfidf_matrix

@st.cache_data
def build_user_item_matrix(ratings, movies):
    merged = ratings.merge(movies[['movieId']], on='movieId')
    user_item = merged.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    return user_item

def svd_recommendation(user_item_matrix, user_id, svd_model, scaler, movies, top_n=10):
    if user_id not in user_item_matrix.index:
        return None
    
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
    svd_user_vector = svd_model.transform(user_vector)
    svd_reconstruction = svd_model.inverse_transform(svd_user_vector)
    svd_scores = scaler.fit_transform(svd_reconstruction.reshape(-1, 1)).flatten()

    movie_ids = user_item_matrix.columns
    df_scores = pd.DataFrame({'movieId': movie_ids, 'svd_score': svd_scores})

    rated_movie_ids = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index.tolist()
    
    unrated_df = df_scores[~df_scores['movieId'].isin(rated_movie_ids)]
    
    recommendations = unrated_df.sort_values('svd_score', ascending=False).head(top_n)
    recommendations = recommendations.merge(movies[['movieId', 'title', 'genres']], on='movieId')
    return recommendations


def content_recommendation(movie_title, movies, tfidf_matrix, top_n=10):
    if movie_title not in movies['title'].values:
        return None
    
    idx = movies.index[movies['title'] == movie_title][0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]

def evaluate_recommendation(ratings, user_item_matrix, svd_model):
    from sklearn.model_selection import train_test_split
    
    user_item = user_item_matrix.copy()
    ratings_list = []
    for user in user_item.index:
        for movie in user_item.columns:
            if user_item.loc[user, movie] > 0:
                ratings_list.append((user, movie, user_item.loc[user, movie]))
    ratings_df = pd.DataFrame(ratings_list, columns=['userId', 'movieId', 'rating'])
    
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    
    train_matrix = train_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    
    train_user_item = train_matrix.values
    svd_model.fit(train_user_item)
    
    y_true = []
    y_pred = []
    
    for _, row in test_df.iterrows():
        user_idx = train_matrix.index.get_loc(row['userId']) if row['userId'] in train_matrix.index else None
        movie_idx = train_matrix.columns.get_loc(row['movieId']) if row['movieId'] in train_matrix.columns else None
        if user_idx is not None and movie_idx is not None:
            user_vec = train_user_item[user_idx].reshape(1, -1)
            svd_user_vec = svd_model.transform(user_vec)
            svd_reconstructed = svd_model.inverse_transform(svd_user_vec)
            score = svd_reconstructed[0][movie_idx]
            y_true.append(row['rating'])
            y_pred.append(score)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)
    mae = mean_absolute_error(y_true, y_pred)
    
    y_true_bin = [1 if r >= 4 else 0 for r in y_true]
    y_pred_bin = [1 if p >= 4 else 0 for p in y_pred]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average='binary', zero_division=0)
    
    return rmse, mae, precision, recall, f1

st.title("🎬 Movie Recommendation App")

movies, ratings = load_data()
movies = preprocess_movies(movies)
tfidf_matrix = build_tfidf_matrix(movies)
user_item_matrix = build_user_item_matrix(ratings, movies)

svd_model = TruncatedSVD(n_components=20, random_state=42)
svd_model.fit(user_item_matrix.values)
scaler = MinMaxScaler()

st.sidebar.header("User Input")

user_id = st.sidebar.number_input("Enter your User ID:", min_value=1, max_value=int(user_item_matrix.index.max()), value=1)

movie_title = st.sidebar.selectbox("Select a movie you like:", movies['title'].sample(50).values)

if st.sidebar.button("Get Recommendations"):
    st.subheader("Content-Based Recommendations (Similar Movies):")
    content_recs = content_recommendation(movie_title, movies, tfidf_matrix)
    if content_recs is not None:
        st.dataframe(content_recs)
    else:
        st.error("Movie not found!")

    st.subheader(f"SVD-based Recommendations for User {user_id}:")
    svd_recs = svd_recommendation(user_item_matrix, user_id, svd_model, scaler, movies)
    if svd_recs is not None:
        st.dataframe(svd_recs[['title', 'genres', 'svd_score']])
    else:
        st.error("User ID not found or has no recommendations!")

    st.subheader("Evaluation Metrics:")
    rmse_val, mae_val, prec, rec, f1 = evaluate_recommendation(ratings, user_item_matrix, svd_model)
    st.write(f"RMSE: {rmse_val:.3f}")
    st.write(f"MAE: {mae_val:.3f}")
    st.write(f"Precision: {prec:.3f}")
    st.write(f"Recall: {rec:.3f}")
    st.write(f"F1-score: {f1:.3f}")
