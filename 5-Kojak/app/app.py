from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads,IMAGES
from scipy.misc import imsave, imread, imresize
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

import keras.models

import re
import sys
import os
sys.path.append(os.path.abspath("./model"))

from load import *

app = Flask(__name__)

global model,graph

photos = UploadSet('photos',IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = '.'
configure_uploads(app,photos)

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/upload',methods=['GET', 'POST'])
def upload():
	if request.method == 'POST' and 'photo' in request.files:
		filename = photos.save(request.files['photo'])
		os.rename('./'+filename,'./'+'output.png')

		print("debug")
		#read the image into memory
		img = image.load_img('./output.png', target_size = (224,224))

		img_data = image.img_to_array(img)
		img_data = np.expand_dims(img_data,axis=0)
		img_data = preprocess_input(img_data)

		pred_class = idx2class[model.predict_classes(img_data)[0]]

		return render_template("index2.html",cl = pred_class)

if __name == "__main__":
	port = int(os.environ.get('PORT', 5000))

	app.run(host='0.0.0.0',port = port)
