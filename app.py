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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def get_Predictions(filename):
    for file in os.listdir('static/segmented_images'):
        os.remove(f'static/segmented_images/{file}')
    
    for file in os.listdir('static/crop'):
        os.remove(f'static/crop/{file}')
    
    seg_model=load_model("static/models/segmentation_model.h5") 
    class_names = ['Flat', 'Gable', 'Hip']
    model = load_model('static/models/classification_model.h5')
    
    img=load_img(filename,target_size=(256,256))
    img=img_to_array(img)
    img = img/255
    test_img_norm=img[:,:,0][:,:,None]
    test_img_input=np.expand_dims(test_img_norm, 0)
    prediction = (seg_model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    prediction_save = Image.fromarray((prediction * 255))
    prediction_save.save('static/segmented_images/2.png')

    segmentation_plot = color.label2rgb(prediction, img, kind="overlay", saturation=1)
    
    im = Image.fromarray((segmentation_plot * 255).astype(np.uint8))
    

    im.save('static/segmented_images/1.png')

    image=cv2.imread('static/segmented_images/2.png')
    mainimage=cv2.imread(f'static/requested_images/{filename}')
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    kernel=np.ones((3,3),np.uint8)
    dilated=cv2.dilate(th2,kernel,iterations=3)
    
    contours,_ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_box=[cv2.boundingRect(cnt) for cnt in contours]
    i = 0
    areas = []
    classes = []
    location = []
    print(len(contours_box))
    for cnt in contours:
        x,y,w,h=cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area >300:
            box_image = mainimage[y : y+h, x: x+w]
            if(i!=0):
                areas.append(area)
                cv2.imwrite(f'static/crop/{i}_{area}.jpg', box_image)
                img=load_img(f'static/crop/{i}_{area}.jpg',target_size=(150,150))
                img=img_to_array(img)
                img = img/255
                proba = model.predict(img.reshape(1,150,150,3))
                top_3 = np.argsort(proba[0])[:-4:-1]
                topclasses = []
                for j in range(3):
                    topclasses.append(class_names[top_3[j]])
                classes.append(topclasses[0])
                location.append(f'static/crop/{i}_{topclasses[0]}_{area}.jpg')
                os.remove(f'static/crop/{i}_{area}.jpg')
                cv2.imwrite(f'static/crop/{i}_{topclasses[0]}_{area}.jpg', box_image)
            i=i+1        
    return location, classes, areas
    
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


app=Flask(__name__, static_url_path='/static')


@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/',methods=['POST', 'GET'])
def submit():
    logging.info("done")

    image = request.files["img"]
    scale = (request.form['scale'])
    name= image.filename
    for file in os.listdir('static/requested_images'):
        os.remove(f'static/requested_images/{file}')
    image.save(f'static/requested_images/{name}')

    cropped, labels, main_areas = get_Predictions(f'static/requested_images/{name}')
    
    # cropped = ['static/crop/1_Flat_4347.0.jpg', 'static/crop/2_Flat_1170.5.jpg', 'static/crop/3_Flat_379.5.jpg', 'static/crop/4_Flat_3824.5.jpg']
    # labels = ['Flat', 'Flat', 'Flat', 'Flat'] 
    # main_areas = [4347.0, 1170.5, 379.5, 3824.5] 

    areas = [round(element * int(scale), 2) for element in main_areas]
    main_image = f'static/requested_images/{name}'
    seg_img = f'static/segmented_images/1.png'
    return render_template('index.html',image=main_image, seg_img = seg_img, cropped = cropped, labels=labels, area = areas) 

    
if __name__=='__main__':
    port = os.environ.get("PORT",5000)
    app.run(debug=False,host='0.0.0.0',port=port)
