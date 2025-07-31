# app/main.py

from flask import Flask, request, render_template
import joblib
import numpy as np
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "../model/iris_model.pkl")
model = joblib.load(model_path)

app = Flask(__name__, template_folder="templates")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        features = [float(request.form[f]) for f in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        prediction = model.predict([features])
        species = ['Setosa', 'Versicolor', 'Virginica'][prediction[0]]
        return render_template("index.html", prediction_text=f"Predicted Iris Species: {species}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
