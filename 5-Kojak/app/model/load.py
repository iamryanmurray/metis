import numpy as np
import keras.models
from keras.models import load_model
from scipy.misc import imread, imresize,imshow
import pickle
import tensorflow as tf
from keras import optimizers


def init():
	loaded_model = load_model("22_models_dropout_continue.h5")
	print("Loaded Model from disk")

	with open('idx2class.pkl','rb') as pf:
		idx2class = pickle.load(pf)
	print("Loaded idx2class")
	loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

	graph = tf.get_default_graph()
	
	return loaded_model,idx2class,graph
