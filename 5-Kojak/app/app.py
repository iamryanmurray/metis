from flask import Flask, render_template, request, send_from_directory
from flask_uploads import UploadSet, configure_uploads,IMAGES
from scipy.misc import imsave, imread, imresize
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

import keras.models

import re
import sys
import os
import pickle
sys.path.append(os.path.abspath("./model"))

from load import *

app = Flask(__name__)

global model ,idx2class,graph


model,idx2class,graph = init()

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
		with graph.as_default():
			global pred_class
			pred_class = idx2class[model.predict_classes(img_data)[0]]
		

			return render_template("index2.html",cl = pred_class)

@app.route('/success',methods=['GET','POST'])
def correct_prediction():
	if request.method == 'POST':
		os.rename('./output.png', './images/'+ pred_class +'_{0}.png'.format(np.random.randint(100000)))
		return render_template('success.html')
	return render_template('success.html')

@app.route('/failure',methods=['GET','POST'])
def incorrect_prediction():
	if request.method == 'POST':
		return render_template('failure.html')



@app.route('/label',methods=['GET','POST'])
def label_image():
	make = request.form['make'].lower()
	model = request.form['model'].lower()
	os.rename('./output.png','./images/'+make+'_'+model+'_{0}.png'.format(np.random.randint(1000000)))
	return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route("/<string:page_name>")
def render_static(page_name):
	return render_template('%s' % page_name)

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))

	app.run(host='0.0.0.0',port = port)
	app.run(debug=True)
