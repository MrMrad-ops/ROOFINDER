from flask import Flask,render_template,request
import pickle
import numpy as np
import os
import tensorflow as tf
from keras.models import load_model
import cv2
# hide TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def get_seg_Prediction(filename):
        
   
    my_seg_model=load_model("seg_model\full_best_model.h5")  #Load model
    
    SIZE = 256 #Resize 
    img_path = 'seg_model\images'+filename
    img = np.asarray(cv2.resize(cv2.imread(img_path, 0),(SIZE,SIZE)))
    
    img = np.expand_dims(img, axis=3)        
    img = img/255.
    pred = my_seg_model.predict(img) # segmentation                    
    pred = pred > 0.5  #binary image
    cv2.imwrite(os.path.join(img_path , 'segmented.jpg'),pred) #save the segmented image
    return pred

def measure(prediction):
    ret1, thresh = cv2.threshold(prediction, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.25*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    ret3, markers = cv2.connectedComponents(sure_fg)
    props = measure.regionprops_table(markers, intensity_image=prediction, 
                              properties=['label',
                                          'area', 'equivalent_diameter',
                                          'mean_intensity', 'solidity','orientation'])
    return props


app=Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

#@app.route('/',methods=['POST','GET'])
#def submit():



  #  return    
    
if __name__=='__main__':
    port = os.environ.get("PORT",5000)
    app.run(debug=False,host='0.0.0.0',port=port)
