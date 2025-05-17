from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
from flask_cors import CORS
import csv
from datetime import datetime
import os
import cv2
from PIL import Image
import io
import base64

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

def preprocess_image(image_data):
    """
    Preprocess the image to match our model's format:
    - Convert to grayscale
    - Resize to 8x8 (64 features)
    - Normalize pixel values
    - Invert colors if needed
    """
    # Convert base64 to image
    if isinstance(image_data, str) and image_data.startswith('data:image'):
        # Remove the data URL prefix
        image_data = image_data.split(',')[1]
    
    # Decode base64 image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Add padding to make the image square
    height, width = img_array.shape
    size = max(height, width)
    pad_h = (size - height) // 2
    pad_w = (size - width) // 2
    img_array = np.pad(img_array, ((pad_h, size - height - pad_h), (pad_w, size - width - pad_w)), mode='constant', constant_values=0)
    
    # Resize to 8x8
    img_array = cv2.resize(img_array, (8, 8), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to 0-16 range (like our training data)
    img_array = (img_array / 255.0 * 16.0).astype(np.float32)
    
    # Invert colors if needed (our model expects white digits on black background)
    if img_array.mean() > 8:  # If background is white (assuming 16 is max value)
        img_array = 16 - img_array
    
    # Flatten the image to 1D array of 64 pixels
    img_array = img_array.reshape(1, -1)
    
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request contains drawn pixels or an uploaded image
        if 'image' in request.json:
            # Process uploaded image
            pixels = preprocess_image(request.json['image'])
        else:
            # Process drawn pixels
            data = request.json['pixels']
            # Convert drawn pixels to 8x8 image
            img_array = np.array(data, dtype=float).reshape(8, 8)
            
            # Normalize pixel values to 0-16 range if they aren't already
            if img_array.max() > 16:
                img_array = (img_array / 255.0 * 16.0).astype(np.float32)
            
            # Invert colors if needed (our model expects white digits on black background)
            if img_array.mean() > 8:  # If background is white (assuming 16 is max value)
                img_array = 16 - img_array
            
            # Reshape to 1x64 for prediction
            pixels = img_array.reshape(1, -1)
        
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