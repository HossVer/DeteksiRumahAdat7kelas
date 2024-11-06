#-------------------------------------------------------------------------------------------
#            Script ini akan mengembalikan kelas dari citra rumah adat                     #
#-------------------------------------------------------------------------------------------
import keras
##--Import Library
import numpy as np
import tensorflow as tf
from keras.api.models import load_model
from PIL import Image

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

##--Klasifikasi funct
def classify(image_dir):
    ##--Load model dan gambar
    model = load_models()
    image = load_image(image_dir)

    ##--Preprocessing
    ##--Rezise
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
    print("Predicted class: ", prediction_class)


classify("../image/honai (33).jpg")
