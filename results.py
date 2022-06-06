import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from skimage import color
from PIL import Image
import keras
import cv2
import os
import solar_panels_area

def get_Predictions(filename, scale):
    for file in os.listdir('static/segmented_images'):
        os.remove(f'static/segmented_images/{file}')
    
    for file in os.listdir('static/crop'):
        os.remove(f'static/crop/{file}')
    
    for file in os.listdir('static/panels'):
        os.remove(f'static/panels/{file}')   
    
    seg_model=load_model("static/models/400-full_best_model.h5") 
    class_names = ['Flat', 'Gable', 'Hip']
    model = keras.models.load_model('static/models/finalmodel.h5')
    
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
    
    seg_image = 'static/segmented_images/1.png'
    im.save(seg_image)

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
    cropped = []
    panels = []
    for cnt in contours:
        x,y,w,h=cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area >300:
            if(i!=0):
                box_image = mainimage[y : y+h, x: x+w]
                # areas.append(area)
                cv2.imwrite(f'static/crop/{i}.jpg', box_image)
                img=load_img(f'static/crop/{i}.jpg',target_size=(150,150))
                img=img_to_array(img)
                img = img/255
                proba = model.predict(img.reshape(1,150,150,3))
                top_3 = np.argsort(proba[0])[:-4:-1]
                topclasses = []
                for j in range(3):
                    topclasses.append(class_names[top_3[j]])
                classes.append(topclasses[0])
                cropped.append(f'static/crop/{i}.jpg')
                os.remove(f'static/crop/{i}.jpg')
                cv2.imwrite(f'static/crop/{i}.jpg', box_image)

                area, panel = solar_panels_area.draw_panels_area(f'static/crop/{i}.jpg', f'{i}.jpg')
                areas.append(area)
                panels.append(panel)
            i=i+1       
    roofareas = [round(element * int(scale), 2) for element in areas] 
    return seg_image, cropped, classes, roofareas, panels