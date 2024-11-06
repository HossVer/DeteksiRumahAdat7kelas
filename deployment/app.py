#-------------------------------------------------------------------------------------------
#            Script ini akan mengembalikan kelas dari citra rumah adat                     #
#-------------------------------------------------------------------------------------------

##--Import Library
import os
import numpy as np
import keras as keras
import requests
import tensorflow as tf
from keras.api.models import load_model
from PIL import Image
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request

##--Instantiate Flask App
app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(["jpg","png","jpeg"])
app.config['UPLOAD_FOLDER'] = '../statics/image'


##------------------PYTHON FUNCTION-----------------------------------
##--Klasifikasi funct
def classify(image_dir):
    ##--Load model dan gambar
    model = load_models()
    image = load_image(image_dir)

    ##--Preprocessing
    ##--Resize
    image = image.resize((224,224))

    ##--Convert into numpy array and convert to tensor array
    img_array = np.asarray(image).astype('uint8')
    img_array = tf.expand_dims(img_array, 0)

    ##--Predict dan ambil nilai konfidensi tertinggi
    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0])

    ##--Set the image class
    label = load_label()
    prediction_class = label[np.argmax(score)]
    return prediction_class, score

##--Load labels
def load_label():
    labels = ['balai', 'bukanrudat', 'gadang', 'honai', 'joglo', 'panjang', 'rumah bali', 'tongkonan']
    return labels

##--Load Model Klasifikasi funct
def load_models():
    model = load_model("../classification_model/cnn_modelll.h5")
    return  model

##--load image funct
def load_image(image_dir):
    image = Image.open(image_dir).convert('RGB')
    return image

##--Check image extension 
def allowed_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

##=--Check server availability
def check_server_availability(destination_url, timeout=30):
    try:
        response = requests.get(destination_url, timeout=timeout)
        if response.status_code == 400:
            return True
        else:
            return False
    except requests.exceptions.Timeout:
        return False

##---------------------------FLASK ROUTING---------------------------------
@app.route("/", methods=['GET'])
def homepage():
    return jsonify({
        "data": None,
        "status": {
            "code":200,
            "message": "deteksi rumah adat indonesia API is running"
        },}
    ),200

@app.route("/prediction", methods=['GET','POST'])
def prediction():
    ##--Get the uploaded image
    if request.method == 'POST':
        image = request.files['image']
        if image and allowed_extension(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            ##--Get uploaded image path
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            ##--Classify image and get the score
            prediction, score = classify(image_path)
            
            ##--Return the results
            return jsonify({
                "data": {
                    "class_name":prediction,
                    "confidence_score":score,
                },
                "status": {
                    "code":200,
                    "message": "success classifying rumah adat"
                },
           }),200
        else:
            return jsonify({
                "data":None,
                "status": {
                    "code" : 503,
                    "message": "failed to fetch data, please try again"
                },
            }),503
    else:
        return jsonify({
            "data":None,
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
        }),405

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 8080)))