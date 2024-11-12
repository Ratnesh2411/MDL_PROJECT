import os 
import sys
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np 
from PIL import Image
from google.cloud import storage

# Set environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask app with explicit path
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# Load model with error handling
try:
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classifier/cifar10_model.h5')
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded", 500
        
    if 'image' not in request.files:
        return "No file uploaded", 400
        
    file = request.files['image']
    image = Image.open(file).resize((32, 32))
    image = np.array(image).reshape((1, 32, 32, 3)) / 255.0
    predictions = model.predict(image)
    class_idx = np.argmax(predictions[0])
    class_label = labels[class_idx]
    confidence = float(predictions[0][class_idx])
    
    return render_template('index.html', prediction=class_label, confidence=confidence)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
