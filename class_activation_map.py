from __future__ import print_function,division
from builtins import range,input
from keras.models import Model
from keras.applications.resnet50 import Resnet50,preprocess_input,decode_predictions
from keras.preprocessing import image

import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt

from glob import glob

image_files=glob('caltech101/*/*.jp*g')

plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()


resnet=Resnet50(input_shape=(224,224,3),weights='imagenet',include_top=True)


resnet.summary()

activation_layer=resnet.get_layer('activation_49')

model=Model(inputs=resnet.input,outputs=activation_layer.output)

final_dense=renet.get_layer('fc1000')

W=final_dense.get_weights()[0]

while True:
	img=image.load_img(np.random.choice(image_files),target=(224,224))
	x=preprocess_input(np.expand_dims(img,0))
	fmaps=model.predict(x)[0]


	probs=resnet.predict(x)
	classnames=decode_predictions(probs)[0]
	print(classnames)
	classname=classnames[0][1]
	pred=np.argmax(probs[0])

	w=W[:,pred]

	cam=fmaps.dot(w)
	cam=sp.ndimage.zoom(cam,(32,32),order=1)

	plt.subplot(1,2,1)
	plt.imshow(img,alpha=0.8)
	plt.imshow(cam,cmap='jet',alpha=0.5)
	plt.subplot(1,2,2)
	plt.imshow(img)
	plt.title(classname)
	plt.show()

	ans=input("continue?(Y/n")
	if ans and ans[0].lower()=='n':
		break