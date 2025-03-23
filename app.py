from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and scaler (ensure these files are in the same directory as app.py)
kmeans_model = joblib.load('kmeans_model.pkl')  # Load the pre-trained KMeans model
scaler = joblib.load('scaler.pkl')  # Load the pre-trained scaler

# Cluster labels: Modify this to match your actual cluster analysis
cluster_labels = {
    0: "High-income, low spending",
    1: "Low-income, low spending",
    2: "Middle-income, high spending",
    3: "High-income, high spending",
    4: "Low-income, middle spending"
}

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            # Get form data
            age = int(request.form['age'])
            income = int(request.form['income'])
            spending = int(request.form['spending'])

            # Prepare the input data for prediction (scale it)
            customer_data = pd.DataFrame([[age, income, spending]], columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
            customer_data_scaled = scaler.transform(customer_data)

            # Predict the cluster for the new customer
            predicted_cluster = kmeans_model.predict(customer_data_scaled)[0]
            
            # Map the predicted cluster number to a label
            prediction_label = cluster_labels.get(predicted_cluster, "Unknown Cluster")
            prediction = f"Cluster {predicted_cluster}: {prediction_label}"

        except Exception as e:
            error = f"An error occurred: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
