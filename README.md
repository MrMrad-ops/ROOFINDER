# ROOFINDER

# Roofinder Project

This project is to automate the assessment process of solar panel installation, which includes finding available space (rooftops) and specifying the features of that place.


## General pipeline for this project:
* Automatically detect rooftops from satellite images.
* Specify the type of the rooftop (flat, hip, gable). 
* Calculate the surface area excluding obstructions.
* Place suggested solar panels taking into consideration the roof edges, obstacles, and rotation.



## Project Development:

Our approach combines 2 CNN models to segment and classify rooftops in addition to image processing techniques to crop the satellite image based on the segmentation results then calculate the area for each rooftop detected then finally plot suggested placements for solar panels on the detected rooftops.

### CNN Models:
-  U-net model for semantic segmentation to label the rooftops
-  Multi-class classification model to classify roof types (Hip, gable, flat)

### Image processing:
-  Extract each detected rooftop into a separate cropped image.
-  Calculate the area of the roof
-  Plot suggested placements for the solar panel


## Deployment

We built a webpage using Flask, where the user enters the satellite image and the scale of the image (to convert from pixel to sqm). 
Then we containerized the app using docker and deployed it on Heruko using AWS EC2 instance.

<img src="https://user-images.githubusercontent.com/91887942/172738203-390817e9-0f6f-4f6f-977e-202b12f05a22.png" width="700" height="400">



Project Report: https://docs.google.com/document/d/1zOV_ufTdYVlAyo5UlTS2zegPGU7uaHbSB-xvuSI-8cA/edit
