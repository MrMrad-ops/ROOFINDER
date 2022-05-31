from flask import Flask,render_template,request
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# hide TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


app=Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/',methods=['POST','GET'])
def submit():
    
    


if __name__=='__main__':
    port = os.environ.get("PORT",5000)
    app.run(debug=False,host='0.0.0.0',port=port)
