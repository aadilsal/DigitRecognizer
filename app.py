from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
from flask_cors import CORS
import csv
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = load('digits_model.joblib')

# Configure feedback storage (optional)
ENABLE_FEEDBACK = os.environ.get('ENABLE_FEEDBACK', 'false').lower() == 'true'
FEEDBACK_DIR = os.environ.get('FEEDBACK_DIR', 'feedback')

if ENABLE_FEEDBACK:
    # Ensure feedback directory exists
    if not os.path.exists(FEEDBACK_DIR):
        os.makedirs(FEEDBACK_DIR)

    # Initialize CSV file if it doesn't exist
    FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, 'wrong_predictions.csv')
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'predicted_digit', 'pixel_values'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the pixel data from the request
        data = request.json['pixels']
        
        # Convert to numpy array and reshape for prediction
        pixels = np.array(data, dtype=float).reshape(1, -1)
        
        # Make prediction
        prediction = int(model.predict(pixels)[0])
        
        # Get prediction probabilities for all classes
        probabilities = model.predict_proba(pixels)[0]
        
        # Create list of (digit, probability) pairs and sort by probability
        predictions = [(int(digit), float(prob)) for digit, prob in enumerate(probabilities)]
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3 predictions
        top_3 = predictions[:3]
        
        return jsonify({
            'prediction': prediction,
            'confidence': round(float(probabilities[prediction]) * 100, 2),
            'top_3': [
                {'digit': p[0], 'probability': round(p[1] * 100, 2)}
                for p in top_3
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/feedback', methods=['POST'])
def record_feedback():
    if not ENABLE_FEEDBACK:
        return jsonify({
            'status': 'error',
            'message': 'Feedback collection is disabled'
        }), 400

    try:
        data = request.json
        pixels = data.get('pixels', [])
        predicted_digit = data.get('prediction')
        
        # Save to CSV
        timestamp = datetime.now().isoformat()
        with open(FEEDBACK_FILE, 'a', newline='') as f:
            writer = csv.writer(f) 
            writer.writerow([timestamp, predicted_digit, ','.join(map(str, pixels))])
        
        return jsonify({
            'status': 'success',
            'message': 'Thank you for your feedback! This helps us improve the model.'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 