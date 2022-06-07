from flask import Flask,render_template,request, url_for
import numpy as np
from skimage import color
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import cv2
import logging
import solar_panels_area
import results

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
UPLOAD_FOLDER = 'static/requested_images'

app=Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/',methods=['POST', 'GET'])
def submit():
    logging.info("done")

    image = request.files["img"]
    name= image.filename
    #mainimage = f'static/requested_images/{name}'
    #mainimage = image.open(f'static/requested_images/{name}'
    scale = (request.form['scale'])
    
    for file in os.listdir('static/requested_images'):
        os.remove(f'static/requested_images/{file}')
    
    image.save(os.path.join(app.config['UPLOAD_FOLDER'],name))
    #image.save(f'static/requested_images/{name}')
    
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], name)

    
    seg_image, cropped, labels, areas, panels = results.get_Predictions(full_filename, scale)

    # cropped = ['static/crop/1.jpg', 'static/crop/4.jpg', 'static/crop/2.jpg', 'static/crop/3.jpg']
    # labels = ['Hip', 'Hip', 'Flat', 'Flat'] 
    # areas = [91.725, 148.65, 27.599999999999998, 22.575] 
    # panels = ['static/panels/111.jpg', 'static/panels/114.jpg', 'static/panels/112.jpg', 'static/panels/113.jpg']
    # main_image = f'static/requested_images/{name}'
    # seg_img = f'static/segmented_images/1.png'


    return render_template('index.html',image=image, seg_img = seg_image, cropped = cropped, labels=labels, area = areas, panels=panels) 

    
if __name__=='__main__':
    port = os.environ.get("PORT",5000)
    app.run(debug=False,host='0.0.0.0',port=port)
