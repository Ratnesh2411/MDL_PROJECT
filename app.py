# Import standard libraries
import os
import sys

# Import Flask for web application, request handling, and rendering templates
from flask import Flask, request, jsonify, render_template

# Import TensorFlow for loading and running the model
import tensorflow as tf

# Import NumPy for numerical operations
import numpy as np 

# Import PIL for image processing
from PIL import Image

# Import Google Cloud Storage client for accessing cloud resources (if needed)
from google.cloud import storage

# Set environment variables to disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize the Flask application with explicit path to the 'templates' directory
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# Try to load the pre-trained model from the specified path with error handling
try:
    # Define the path to the saved model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classifier/cifar10_model.h5')
    
    # Load the model using TensorFlow
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    # If there's an error, print it and set the model to None
    print(f"Error loading model: {e}")
    model = None

# Define class labels for CIFAR-10 dataset predictions
labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Define a route for the homepage
@app.route('/')
def home():
    # Render the 'index.html' template on the homepage
    return render_template('index.html')

# Define a route to handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the model is loaded; if not, return a server error
    if model is None:
        return "Model not loaded", 500
        
    # Check if an image file is uploaded; if not, return a client error
    if 'image' not in request.files:
        return "No file uploaded", 400
        
    # Get the uploaded image file from the request
    file = request.files['image']
    
    # Open the image using PIL, resize it to 32x32 pixels, and convert to a NumPy array
    image = Image.open(file).resize((32, 32))
    image = np.array(image).reshape((1, 32, 32, 3)) / 255.0  # Normalize the image
    
    # Make a prediction using the loaded model
    predictions = model.predict(image)
    
    # Get the index of the class with the highest predicted probability
    class_idx = np.argmax(predictions[0])
    
    # Get the corresponding class label from the labels list
    class_label = labels[class_idx]
    
    # Get the confidence score of the predicted class
    confidence = float(predictions[0][class_idx])
    
    # Render the 'index.html' template with prediction results
    return render_template('index.html', prediction=class_label, confidence=confidence)

# Run the Flask application
if __name__ == '__main__':
    # Set the port to the environment variable 'PORT' or default to 8080
    port = int(os.environ.get('PORT', 8080))
    
    # Start the Flask server on all available network interfaces
    app.run(host='0.0.0.0', port=port)
