#References the following git hub code: https://github.com/papiot/CarND-Behavioral-Cloning/blob/master/assig.py

#import libraries

import csv
import cv2

import numpy as np
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt


def load_data(args):
  lines = []

  #read lines from log file
  with open(os.path.join(args.data_dir, 'driving_log.csv')) as csvfile:
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
      image_rgb = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      images.append(image_rgb)

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

  X_train = np.array(augmentated_images)
  y_train = np.array(augmentated_measurements)

  return X_train, y_train


def train_model(args, X_train, y_train):
  #import dependencies for training
  from keras.models import Sequential
  from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
  from keras.layers.convolutional import Convolution2D
  from keras.layers import Conv2D
  from keras.layers.pooling import MaxPooling2D

  model = Sequential()
  model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=X_train[0].shape))
  model.add(Lambda(lambda x: (x / 255.0) - 0.5))
  model.add(Dropout(args.keep_prob))
  model.add(Conv2D(24, (5, 5), activation="relu", padding='same'))
  model.add(MaxPooling2D())
  model.add(Conv2D(36, (5, 5), activation="relu", padding='same'))
  model.add(MaxPooling2D())
  model.add(Conv2D(48, (5, 5), activation="relu", padding='same'))
  model.add(MaxPooling2D())
  model.add(Dropout(args.keep_prob))
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

  model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
  history = model.fit(X_train, y_train, validation_split=args.test_size, shuffle=True, epochs=args.epochs, batch_size=args.batch_size)

  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

  # Visualization of model
  from keras.utils.vis_utils import plot_model
  plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

  

  model.save('model-'+datetime.now().strftime("%Y%m%d%H%M%S")+'.h5')


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.2)
    parser.add_argument('-n', help='number of epochs',      dest='epochs',          type=int,   default=10)
    #parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=80)
    #parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    train_model(args, *data)


if __name__ == '__main__':
    main()