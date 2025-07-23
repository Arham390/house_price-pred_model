from flask import Flask, request, jsonify , render_template 
import joblib
import numpy as np
import pandas as pd
import json

from source.rec import recommend
from source.rec import recommend_svd_no_filter  # Add this import



app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load feature column names (from training)
with open("columns.json", "r") as f:
    model_columns = json.load(f)
    
@app.route('/')
def home():
    return render_template('index.html')    


@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json(force=True)
    input_df = pd.DataFrame(data)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    input_scaled = scaler.transform(input_df)
    predictions = model.predict(input_scaled)
    predictions = np.expm1(predictions)

    # Recommend based on prediction
    recommendations = recommend_svd_no_filter(input_df)

    return jsonify({
        "predictions": predictions.tolist(),
        "recommendations": recommendations
    })




if __name__ == '__main__':
    app.run(debug=True)
