import numpy as np
import keras.models
from keras.models import load_model
from scipy.misc import imread, imresize,imshow
import pickle

def init():
	model = load_model("model.h5")
	print("Loaded Model from disk")

	with open('','rb') as pf:
		idx2class = pickle.load(pf)


	
	return loaded_model,idx2class