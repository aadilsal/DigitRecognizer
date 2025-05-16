# Digit Recognizer Web App

A Flask web application that uses a pre-trained RandomForest model to recognize handwritten digits. The model was trained on the sklearn digits dataset and expects 8x8 grayscale images.

## Features

- Interactive 8x8 grid for drawing digits
- Real-time prediction with confidence score
- Sample digit generation
- Reset functionality
- Modern UI with TailwindCSS

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Draw a digit using your mouse in the 8x8 grid
2. Click "Predict" to see the model's prediction
3. Use "Reset" to clear the grid
4. Try "Generate Sample" to test with a pre-made digit

## Technical Details

- The application uses a RandomForest model trained on the sklearn digits dataset
- Input is processed as a 64-length array (8x8 grid flattened)
- The model file (digits_model.joblib) must be present in the root directory
- TailwindCSS is loaded via CDN for styling 