
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def recommend(pred_price, input_df, listings_path='train.csv'):
    # Load listings
    df = pd.read_csv(listings_path)

    # Price range: 10% below to 50% above predicted price
    lower_line = pred_price - 0.1 * pred_price
    upper_line = pred_price * 1.5

    # Filter by price
    filtered = df[(df["SalePrice"] >= lower_line) & (df["SalePrice"] <= upper_line)].copy()

    if filtered.empty:
        return []

    # Select relevant numeric features for similarity
    features = ["GrLivArea", "BedroomAbvGr", "OverallQual", "OverallCond", "GarageCars", "FullBath"]

    # Drop rows with missing values in these features
    filtered = filtered.dropna(subset=features)

    # Prepare feature vectors
    listing_vectors = filtered[features].values

    # Prepare input vector from input_df
    input_vector = input_df[features].values[0].reshape(1, -1)

    # Normalize both using MinMaxScaler
    scaler = MinMaxScaler()
    all_vectors = np.vstack([input_vector, listing_vectors])
    scaled = scaler.fit_transform(all_vectors)

    input_scaled = scaled[0].reshape(1, -1)
    listings_scaled = scaled[1:]

    # Compute cosine similarity
    similarities = cosine_similarity(input_scaled, listings_scaled)[0]

    # Add to DataFrame
    filtered["similarity"] = similarities

    # Sort by similarity and return top 5
    top5 = filtered.sort_values(by="similarity", ascending=False).head(5)

    return top5[["Id", "SalePrice", "BedroomAbvGr", "GrLivArea", "OverallQual", "similarity"]].to_dict(orient="records")



# svd implementation using sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
# Load the trained model and scaler
import joblib
import json
import pandas as pd
import numpy as np

def recommend_svd_no_filter(input_df, listings_path='train.csv'):
    df = pd.read_csv(listings_path)

    features = ["GrLivArea", "BedroomAbvGr", "OverallQual", "OverallCond", "GarageCars", "FullBath"]
    df = df.dropna(subset=features)

    listing_vectors = df[features].values
    input_vector = input_df[features].values[0].reshape(1, -1)

    scaler = MinMaxScaler()
    all_vectors = np.vstack([input_vector, listing_vectors])
    scaled = scaler.fit_transform(all_vectors)

    svd = TruncatedSVD(n_components=2, random_state=42)
    pipeline = make_pipeline(svd, Normalizer(copy=False))
    reduced = pipeline.fit_transform(scaled)

    input_svd = reduced[0].reshape(1, -1)
    listings_svd = reduced[1:]

    similarities = cosine_similarity(input_svd, listings_svd)[0]
    df["similarity_svd"] = similarities

    top5 = df.sort_values(by="similarity_svd", ascending=False).head(5)
    return top5[["Id", "SalePrice", "BedroomAbvGr", "GrLivArea", "OverallQual", "similarity_svd"]].to_dict(orient="records")

