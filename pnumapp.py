from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__,template_folder='templates')

# Model saved with Keras model.save()


#Load your trained model
model = load_model('model_pneum.h5')
#model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
#print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    
    img = image.load_img(img_path, target_size=(200, 200),color_mode='grayscale') #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    resized_arr=cv2.resize(img,(200,200))
    resized_arr=resized_arr.reshape(-1,200,200,1)

   
    preds = model.predict(resized_arr)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('pindex.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        os.remove(file_path)#removes file from the server after prediction has been returned

        # Arrange the correct return according to the model. 
		# In this model 1 is Pneumonia and 0 is Normal.
        str1 = ' Patient Normal'
        str2 = ' Patient has Pneumonia'
        if (preds[0] == 1):
            return str1
        else:
            return str2
    return None

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
    app.run(debug=True)
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
