
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

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
def recommend_svd_no_filter( pred_price , listings_path='train.csv', target_user_id=10, min_price=0, max_price=float('inf')):
    n_users = 100

#filtred logic
    min_price = pred_price - 0.1 * pred_price
    max_price = pred_price * 1.5
   
   
    df = pd.read_csv(listings_path)

    # Assign synthetic user_id and ratings
    df['user_id'] = np.random.randint(1, n_users + 1, size=len(df))
    df['property_id'] = df['Id']
    df['rating'] = np.random.randint(1, 7, size=len(df))  # Ratings from 1 to 6

    # Create user-item interaction matrix
    ratings_matrix = df.pivot_table(index='user_id', columns='property_id', values='rating', fill_value=0)

    

    # Apply SVD
    svd = TruncatedSVD(n_components=20, random_state=42)
    R = ratings_matrix.values
    U = svd.fit_transform(R)            # User feature matrix
    V = svd.components_.T               # Item feature matrix

    # Get user latent vector
    user_index = ratings_matrix.index.get_loc(target_user_id)
    user_vector = U[user_index]         
    # Shape: (20,)

    
    scores = np.dot(V, user_vector)     # Shape: (num_properties,)
    top_indices = scores.argsort()[::-1]

    # Get property IDs
    top_property_ids = ratings_matrix.columns[top_indices]

    # Get all ranked properties
    ranked_df = df[df['property_id'].isin(top_property_ids)][
        ["Id", "SalePrice", "BedroomAbvGr", "GrLivArea", "OverallQual", "property_id"]
    ].drop_duplicates(subset="Id")

    # Add scores to this DataFrame
    property_score_map = dict(zip(ratings_matrix.columns[top_indices], scores[top_indices]))
    ranked_df["score"] = ranked_df["property_id"].map(property_score_map)

    # Filter by price range
    filtered = ranked_df[(ranked_df["SalePrice"] >= min_price) & (ranked_df["SalePrice"] <= max_price)]

    # Re-rank top 5 by score
    final_top = filtered.sort_values(by="score", ascending=False).head(5)

    return final_top.drop(columns=["score", "property_id"]).to_dict(orient="records")
