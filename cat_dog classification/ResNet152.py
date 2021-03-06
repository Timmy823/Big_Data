
# coding: utf-8

# # Import Library
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import os
import pickle

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

#from __future__ import print_function
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout,ZeroPadding2D,concatenate,AveragePooling2D,add
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.preprocessing import image
from keras import optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import activations

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers.normalization import BatchNormalization


# # Functions
def save_history(history, fn):
    with open(fn, 'wb') as fw:
        pickle.dump(history.history, fw, protocol=2)

def load_history(fn):
    class Temp():
        pass
    history = Temp()
    with open(fn, 'rb') as fr:
        history.history = pickle.load(fr)
    return history

def Conv_Block(inpt,nb_filter,kernel_size,strides, with_conv_shortcut=False):
    x = Conv2D(nb_filter[0],kernel_size=(1,1),strides=strides,activation='relu',padding='same')(inpt)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(nb_filter[1],kernel_size=(3,3),strides=1,activation='relu',padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Conv2D(nb_filter[2],kernel_size=(1,1),strides=1,activation='relu',padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    if with_conv_shortcut:
        shortcut = Conv2D(nb_filter[2],kernel_size,strides=strides,padding='same')(inpt)
        shortcut = BatchNormalization(axis=1)(shortcut)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,inpt])
        return x


# # Reference

# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = './data/train'
validation_data_dir = './data/validation'

# # Data Generator

# #### Keras針對圖片數量不夠多的問題，也提供了解法：利用ImageDataGenerator，我們可以利用一張圖片，進行若干運算之後，得到不同的圖片。

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')


# # Building a Convolutional Neural Network


# ### ResNet-152

# 判斷RGB是在矩陣中的第幾個元素?
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
    bn_axis = 1
else:
    input_shape = (img_width, img_height, 3)
    bn_axis = 3
    
inpt = Input(input_shape) 
x = ZeroPadding2D((3,3))(inpt)   
# Convolution Net Layer 1~2    
x = Conv2D(64,(7,7),strides=2,padding='valid', activation = 'relu')(x)
x = BatchNormalization(axis=1)(x)
x = MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)

x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3),strides=1,with_conv_shortcut=True)
x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[64,64,256],kernel_size=(3,3),strides=1)

x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3),strides=2,with_conv_shortcut=True)
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[128,128,512],kernel_size=(3,3),strides=1)


x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=2,with_conv_shortcut=True)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[256,256,1024],kernel_size=(3,3),strides=1)

x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3),strides=2,with_conv_shortcut=True)
x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3),strides=1)
x = Conv_Block(x,nb_filter=[512,512,2048],kernel_size=(3,3),strides=1)

x = AveragePooling2D(pool_size=(7,7))(x)
x = Flatten()(x)

x = Dense(1,activation='sigmoid')(x)
model = Model(inpt,x)

sgd = optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])

# ## Training
history = model.fit_generator(train_generator, steps_per_epoch=120, epochs=50, 
                              validation_data=validation_generator, validation_steps=120, verbose=1)

model.save('model_ResNet152.h5')
save_history(history, 'history_ResNet152.bin')

model.summary()
