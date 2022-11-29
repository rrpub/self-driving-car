#References the following git hub code: https://github.com/papiot/CarND-Behavioral-Cloning/blob/master/assig.py

#import libraries

import csv
import cv2

import numpy as np

#read lines from log file


lines = []

with open('./data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

    
#get images and place in arrays

images = []
measurements = []

print("Getting data....")
correction = 0.5
for line in lines:
  for i in range(3):
    source_path = line[i]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
 
    image = cv2.imread(current_path)
    images.append(image)

    measurement = float(line[3])

    if (i == 1):
      measurement = measurement + correction
    
    if (i == 2):
      measurement = measurement - correction

    measurements.append(measurement)


    #augment the images
    
augmentated_images, augmentated_measurements = [], []

for image, measurement in zip(images, measurements): #remove zip?
  augmentated_images.append(image)
  augmentated_measurements.append(measurement)
  augmentated_images.append(cv2.flip(image, 1))
  augmentated_measurements.append(measurement * -1)

#train the model    
    
X_train = np.array(augmentated_images)
y_train = np.array(augmentated_measurements)

#import dependencies for training

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=X_train[0].shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Dropout(0.2))
model.add(Conv2D(24, (5, 5), activation="relu", padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(36, (5, 5), activation="relu", padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(48, (5, 5), activation="relu", padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation="relu", padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation="relu", padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('model.h5')