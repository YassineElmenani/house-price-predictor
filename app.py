from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("house_price_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    city = request.form['City']
    size = float(request.form['Size'])
    bedrooms = int(request.form['Bedrooms'])
    bathrooms = int(request.form['Bathrooms'])
    garage = int(request.form['Garage'])
    garden = int(request.form['Garden'])
    balcony = int(request.form['Balcony'])
    year_built = int(request.form['YearBuilt'])

    # Prepare data for model input (in same order as training)
    data = pd.DataFrame({
        'City': [city],
        'Size_m2': [size],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'Garage': [garage],
        'Garden': [garden],
        'HasBalcony': [balcony],
        'YearBuilt': [year_built]
    })

    # Predict using the model pipeline
    prediction = model.predict(data)[0]

    # Format the result
    price = round(prediction, 2)

    return render_template('index.html', prediction_text=f"Estimated House Price: {price} MAD")

if __name__ == '__main__':
    app.run(debug=True)
