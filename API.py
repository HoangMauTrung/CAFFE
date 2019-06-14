from flask import Flask, render_template, request
from werkzeug import secure_filename
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf 
from PIL import Image
import numpy as np
import flask
import io
import cv2
import os
import pickle
import pandas as pd


app = Flask(__name__)

UPLOAD_FOLDER = 'Save'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def load_Img():
    X_ = []
    df = pd.read_csv(r'dataImage.csv')
    for i in range (0, len(df['name'])):
        img = cv2.imread(os.path.join('compare', df['name'][i]))
        if img is not None:
            img = cv2.resize(img, (224, 224))
            X_.append(img)
    X_ = np.asanyarray(X_)
    return (df,X_)

def load_upload(filename):

    img = cv2.imread(os.path.join('Save', filename))
    if img is not None:
        img  = cv2.resize(img, (224,224))
    X = np.array([img])

    return X

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")
    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    # change chanel RGB -> BGR(cv2)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = np.expand_dims(image, axis=0)
    # return the processed image
    return image

@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    data = {"success": False}
    if request.method == "POST":

        if not 'file' in request.files:
            return flask.jsonify({'error': 'no file'})

        f = request.files['file']
        filename = f.filename
        f.save(os.path.join('Save', secure_filename(filename)))

       
        model = tf.keras.models.load_model('model_predict.h5')
        # img  = Image.open(os.path.join('Save', secure_filename(filename) ))
        # X = prepare_image(img, (224, 224))
        img = cv2.imread(os.path.join('Save', secure_filename(filename) ))
        img = cv2.resize(img, (224, 224))
        X = np.array([img])

        preds = model.predict(X)[0]
        
        (df,X1) = load_Img()
        X1 = model.predict(X1)
        #print(X1)

        X1 = X1-preds
        
        X1 = X1**2
        
        #print(X1.shape)

        X1 = np.sum(X1,axis = 1)
        
        #print(X1.shape)
       
        sort = np.argsort(X1)

        #print(sort)
        result = sort[:10]

        #print(result)
        data = {}
        data["result"] = []
        for i in result:
            if i>0:
                name = {'id' : str(df['id'][i+1]), 'name': str(df['image'][i+1]), 'dislay' : str(df['display_name'][i+1]) }
            else:
                name = {'id' : str(df['id'][i]), 'name': str(df['image'][i]) , 'dislay' : str(df['display_name'][i+1])}

            data["result"].append(name)
        
        del model

        return flask.jsonify(data)
        
    
if __name__ == "__main__":

    app.run('0.0.0.0', 5000,debug = True)