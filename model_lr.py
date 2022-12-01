import os
import glob
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path 
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt 
from datetime import datetime

print (os.listdir( "data/blackboard/"))
data_dir = Path("data/blackboard/")
train_dir = data_dir/'train'
val_dir = data_dir/'val'
test_dir = data_dir/'test'

print(train_dir)
print(val_dir)
print(test_dir)

def load_train():
    left_cases_dir = train_dir/'Left'
    right_cases_dir = train_dir/'Right'
    # Get the list of all the images
    left_cases = left_cases_dir.glob('*.[jp][pn][g]')
    right_cases = right_cases_dir.glob('*.[jp][pn][g]')
    train_data=[]
    train_label=[]
    for img in left_cases:
        train_data.append(img)
        train_label.append('Left')
    for img in right_cases:
        train_data.append(img)
        train_label.append('Right')
    df=pd.DataFrame(train_data)
    df.columns=['images']
    df['labels']=train_label
    df=df.sample(frac=1).reset_index(drop=True)
    return df

train_data=load_train()
train_data.shape 

#data visulisation
plt.bar(train_data['labels'].value_counts().index,train_data['labels'].value_counts().values)
plt.show()

'''
#showcase some randomly selected images from our training data
def plot(image_batch, label_batch):
    plt.figure(figsize=(10,5))
    for i in range (10) :
        ax = plt.subplot(2,5,i+1)
        img = cv2.imread(str(image_batch[i]))
        img = cv2.resize(img, (224,224))
        plt.imshow(img)
        plt.title(label_batch[i])
        plt.axis("off")

plot (image_batch,label_batch)
'''


def prepare_and_load(isval=True):
    if isval==True:
        left_dir = val_dir/'Left'
        right_dir = val_dir/'Right'
    else:
        left_dir = test_dir/'Left'
        right_dir = test_dir/'Right'
    left_cases = left_dir.glob('*.png')
    right_cases = right_dir.glob('*.png')
    
    '''
    if isval==False:
        left_cases = left_dir.glob('*.jpg')
        right_cases = right_dir.glob('*.jpg')
    '''
    
    left_cases = left_dir.glob('*.[jp][pn][g]')
    right_cases = right_dir.glob('*.[jp][pn][g]')
    
    data,labels=([] for x in range(2))
    def prepare (case) :
        for img in case:
            img = cv2.imread(str(img))
            img = cv2.resize(img, (224,224))
            if img.shape[2]==1:
                img = np.dstack([img, img, img])
            img = cv2. cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)/255.
            if case==left_cases:
                label = to_categorical(0, num_classes=2)
            else:
                label = to_categorical(1, num_classes=2)
            data.append(img)
            labels.append(label)
        return data, labels
    prepare(left_cases)
    d,l=prepare(right_cases)
    d=np.array(d)
    l=np.array(l)
    return d,l

val_data, val_labels=prepare_and_load(isval=True)
test_data, test_labels=prepare_and_load(isval=False)
print('Number of test images --> ',len(test_data))
print('Number of validation images --› ',len(val_data))


def data_gen(data, batch_size):
    # Get total number of samples in the data
    n = len(data)
    steps = n//batch_size

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size,2), dtype=np.float32)
    
    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)
    
    # Initialize a counter
    i=0
    while True:
        np.random.shuffle(indices)
        # Get the next batch
        count = 0
        next_batch = indices[(i*batch_size): (i+1)*batch_size]
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['images']
            label = data.iloc[idx]['labels']
            if label=="Left":
                label=0
            else: 
                label=1
            # one hot encoding
            encoded_label = to_categorical(label, num_classes=2)

            # read the image and resize
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224,224))

            # check if it's grayscale
            if img.shape[2]==1:
                img = np.dstack([img, img, img])
            
            # cv2 reads in BR mode by default
            orig_img = cv2. cvtColor(img, cv2.COLOR_BGR2RGB)
            # normalize the image pixels
            orig_img = img.astype(np.float32)/255.
            batch_data[count] = orig_img
            batch_labels[count] = encoded_label
            count+=1
            if count==batch_size-1:
                break
            i+=1
            yield batch_data, batch_labels
            if i>=steps:
                i=0


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation( 'relu'))
model.add(Dense(2))
model.add(Activation('softmax')) 

batch_size = 16
nb_epochs = 25
# Get a train data generator
train_data_gen = data_gen(data=train_data, batch_size=batch_size)
# Define the number of training steps
nb_train_steps = train_data.shape[0]//batch_size
print("Number of training and validation steps: {} and {}".format(nb_train_steps, len(val_data)))

'''
model.compile(loss='binary_crossentropy', optimizer= 'rmsprop', metrics=['accuracy'])
# Fit the model
history = model.fit_generator(train_data_gen, epochs=nb_epochs, steps_per_epoch=nb_train_steps, validation_data=(val_data, val_labels))
'''

def vgg16_model(num_classes=None):
    model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    x=Dense(1024, activation='relu')(model.layers[-4].output) # add my own dense layer after the last conv block
    x=Dropout(0.2)(x)
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.2)(x)
    x=Dense(2, activation='softmax')(x)
    model=Model (model.input,x)
    return model

vgg_conv=vgg16_model(2)
for layer in vgg_conv.layers[:-10]:#freeze all Layers except the Last ten
    layer.trainable = False
opt = RMSprop(learning_rate=0.00001, momentum=0.2)
vgg_conv.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer=opt)

# Fit the model.
history = vgg_conv.fit_generator(train_data_gen, epochs=nb_epochs, steps_per_epoch=nb_train_steps,validation_data=(val_data, val_labels))

loss, acc = vgg_conv.evaluate(test_data, test_labels, batch_size=16)
print('Loss and accuracy', loss, '&',acc) 

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

# Get predictions
pred = vgg_conv.predict(test_data, batch_size=16)
pred = np.argmax(pred, axis=-1)
# Original Labels
labels = np.argmax(test_labels, axis=-1)
from sklearn.metrics import classification_report
print(classification_report(labels, pred)) 

vgg_conv.save('model_lr-'+datetime.now().strftime("%Y%m%d%H%M%S")+'.h5')
