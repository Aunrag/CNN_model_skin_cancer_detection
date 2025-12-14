from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = keras.models.load_model('model.keras')

# Class labels (adjust if needed)
CLASS_NAMES = ['Benign','Melanoma']

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error='No file uploaded')

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', error='No file selected')

    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(image_path)

    # Preprocess image
    img = preprocess_image(image_path)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    result = f"{prediction} {predicted_class} ({confidence:.2f}% confidence)"

    return render_template(
        'index.html',
        prediction=result,
        image_path=image_path
    )

if __name__ == '__main__':
    app.run(debug=True)
