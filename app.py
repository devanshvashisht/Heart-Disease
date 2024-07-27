from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Initialize Flask app
app = Flask(__name__)

CORS(app)


# Load and prepare the model
heartData = pd.read_csv("heart.csv")
X = heartData.drop(columns='target', axis=1)
Y = heartData['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=2)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    data = request.form
    age = float(data['age'])
    gender = float(data['gender'])
    chestPainType = float(data['chestPainType'])
    restingBp = float(data['restingBp'])
    cholestrol = float(data['cholestrol'])
    fastingBloodsugar = float(data['fastingBloodsugar'])
    restingECG = float(data['restingECG'])
    maxHeartRate = float(data['maxHeartRate'])
    exAnigna = float(data['exAnigna'])
    oldPeak = float(data['oldPeak'])
    stSlope = float(data['stSlope'])

    # Prepare the input for the model
    input_data = np.array([[
        age, gender, chestPainType, restingBp, cholestrol, fastingBloodsugar, restingECG,
        maxHeartRate, exAnigna, oldPeak, stSlope
    ]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Send prediction result
    result = 'Heart disease detected' if prediction[0] == 1 else 'No heart disease detected'
    return jsonify({'prediction': result})

# if __name__ == '__main__':
#     app.run(debug=True)
