import csv
import cv2 
import numpy as np
import keras
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import sklearn
from sklearn.model_selection import train_test_split
import random
import math
import os

def get_data(csv_source,img_source,correction):
	# Returns dictionaries of left, center and right images and measurements given source path
	lines=[]

	with open(csv_source) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

	images = {'center':[],'left':[],'right':[]}
	measurements = {'center':[],'left':[],'right':[]}

	for line in lines:	
		for i in range(3):
			source_path = line[i]
			tokens = source_path.split('/')
			filename = tokens[-1]
			local_path = img_source + filename
			image = cv2.imread(local_path)
			measurement = float(line[3])

			if i==0:
				measurements['center'].append(measurement)
				images['center'].append(image)
			elif i==1:
				measurement += correction
				measurements['left'].append(measurement)
				images['left'].append(image)	
			else:
				measurement += -correction
				measurements['right'].append(measurement)
				images['right'].append(image)			

	return images,measurements


def flip_image(image,measurement):
	flipped_image = cv2.flip(image,1)
	flipped_measurement = measurement*-1.0
	return flipped_image, flipped_measurement 


	
folder1 = './Training-Data/Plain-Lap1/'	
images1, measurements1 = get_data(folder1+'driving_log.csv',folder1+'IMG/',0.2)
folder2 = './Training-Data/Plain-Lap2/'	
images2, measurements2 = get_data(folder2+'driving_log.csv',folder2+'IMG/',0.1)		
# folder3 = './Training-Data/Plain-Lap3/'	
# images3, measurements3 = get_data(folder3+'driving_log.csv',folder3+'IMG/',0.2)
# folder4 = './Training-Data/Plain-Lap4/'	
# images4, measurements4 = get_data(folder4+'driving_log.csv',folder4+'IMG/',0.1)
# folder5 = './Training-Data/Plain-Lap5/'	
# images5, measurements5 = get_data(folder5+'driving_log.csv',folder5+'IMG/',0.1)

original_images = []
original_measurements = []
for key in ['left','center','right']:
	original_images.extend(images1[key]+images2[key])
	original_measurements.extend(measurements1[key]+measurements2[key])

del images1, images2
del measurements1, measurements2

# Augment data by flipping/darkening images
transformed_images = []
transformed_measurements = []
for image, measurement in zip(original_images,original_measurements):
	flipped_image, flipped_measurement= flip_image(image,measurement)
	transformed_images.append(flipped_image)
	transformed_measurements.append(flipped_measurement)

	dark_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
	dark_image[:,:,2] = dark_image[:,:,2]*random.uniform(0.25,0.75)
	dark_image = cv2.cvtColor(dark_image, cv2.COLOR_HSV2BGR)
	transformed_images.append(dark_image)
	transformed_measurements.append(measurement)
	

augmented_images = original_images + transformed_images
augmented_measurements = original_measurements + transformed_measurements

# plt.hist(original_measurements,bins=20)
# plt.title("Original Images Steering Angle Histogram")
# plt.xlabel("Class")
# plt.ylabel("Frequency")
# plt.show()

# plt.hist(augmented_measurements,bins=20)
# plt.title("Original + Augmented Steering Angle Histogram")
# plt.xlabel("Class")
# plt.ylabel("Frequency")
# plt.show()

del transformed_images, transformed_measurements, original_images, original_measurements

# Balance dataset 
balanced_images = []
balanced_measurements = []
for i in np.arange(-1, 1, 0.1):
	z = [(x,y) for x,y in zip(augmented_images,augmented_measurements) if (y>=i and y<i+0.1)]
	images, measurements = map(list, zip(*z))
	freq = len(z)
	if freq <1500:
		balanced_images.extend(images)
		balanced_measurements.extend(measurements)
	else:
		r=random.sample(z,1500)
		images, measurements = map(list, zip(*r))
		balanced_images.extend(images)
		balanced_measurements.extend(measurements)

del augmented_images, augmented_measurements

# Split data into training and validation sets
X_train,X_validation,y_train,y_validation = train_test_split(balanced_images,balanced_measurements,test_size=0.2,random_state=0)

# plt.hist(y_train,bins=20)
# plt.title("Training Set Steering Angle Histogram")
# plt.xlabel("Class")
# plt.ylabel("Frequency")
# plt.show()

del balanced_images, balanced_measurements 

def generator(images,angles,batch_size=32):
    num_samples = len(angles)
    while 1:
        for offset in range(0, num_samples, batch_size):
           
            X_chunk = np.array(images[offset:offset+batch_size])
            y_chunk = np.array(angles[offset:offset+batch_size])

            yield sklearn.utils.shuffle(X_chunk, y_chunk)

training_generator = generator(X_train,y_train,batch_size=32)
validation_generator = generator(X_validation,y_validation,batch_size=32)            

			
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation 
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout
from keras.callbacks import EarlyStopping

# Define architecture of CNN model
model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255 - 0.5))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu',))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))


model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
history_object=model.fit_generator(training_generator, samples_per_epoch=len(y_train), validation_data=validation_generator, nb_val_samples=len(y_validation), nb_epoch=3, callbacks=[early_stopping], verbose=1)
model.save('model.h5')


### Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

