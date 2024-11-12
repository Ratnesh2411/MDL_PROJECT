import os 
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np 
from PIL import Image
from google.cloud import storage

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Initialize Flask app
app = Flask(os.name)
model = tf.keras.models.load_model('classifier/cifar10_model.h5')

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

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

if os.name == 'main':
    # Use the PORT environment variable or default to 8000
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)


def download_data():
    client = storage.Client()
    bucket = client.get_bucket('cifar10flaskproject')
    blob = bucket.blob('data/dataset.zip')
    blob.download_to_filename('data/dataset.zip')