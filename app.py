from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
model = load_model(r'artifacts\fruits.keras')

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

fruits_dict = {
    'apple': 0,
    'banana': 1,
    'cherry': 2,
    'chickoo': 3,
    'grape': 4,
    'kiwi': 5,
    'mango': 6,
    'orange': 7,
    'strawberry': 8
}

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route("/detect", methods=['POST'])
def recognize():
    imgfile = request.files['fruit']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], imgfile.filename)
    imgfile.save(image_path)
    
    img = load_img(image_path, target_size=(224, 224))
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    
    pred = model.predict(img_arr)
    score = tf.nn.softmax(pred)
    
    for key, val in fruits_dict.items():
        if val == np.argmax(score):
            msg = f"This is a {key}"
    
    return render_template('index.html', text=msg, img_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
