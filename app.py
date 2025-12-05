from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomHeight, RandomWidth, RandomContrast
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model(
    'big_cats_model.h5',
    custom_objects={
        'RandomFlip': RandomFlip,
        'RandomRotation': RandomRotation,
        'RandomZoom': RandomZoom,
        'RandomHeight': RandomHeight,
        'RandomWidth': RandomWidth,
        'RandomContrast': RandomContrast
    }
)

class_names = ['AFRICAN LEOPARD', 'CARACAL', 'CHEETAH', 'CLOUDED LEOPARD', 'JAGUAR', 'LIONS', 'OCELOT', 'PUMA', 'SNOW LEOPARD', 'TIGER']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    img_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction='No file selected')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction='No file selected')

        # Save uploaded file
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        # Preprocess
        img = load_img(img_path, target_size=(224,224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        prediction = class_names[np.argmax(preds[0])]

    return render_template('index.html', prediction=prediction, img_path=img_path)


if __name__ == '__main__':
    app.run(debug=True)