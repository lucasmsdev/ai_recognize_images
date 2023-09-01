import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from flask import Flask, render_template, request
from PIL import Image  # Importe a classe Image do módulo PIL
app = Flask(__name__)

def preprocess_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

model = tf.keras.applications.ResNet50(weights='imagenet')

@app.route('/', methods=['GET', 'POST'])
def upload_and_analyze():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image = Image.open(image_file)
            input_image = preprocess_image(image)
            predictions = model.predict(input_image)
            decoded_predictions = decode_predictions(predictions, top=3)[0]
            return render_template('result.html', predictions=decoded_predictions)
    return render_template('upload.html')
    
@app.route('/devs')
def devs():
    return render_template('devs.html')

if __name__ == '__main__':
    app.run(debug=True, port=8000)