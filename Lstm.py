# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:38:23 2021

@author: Tom
"""

import numpy as np
import pandas as pd
import re
import math
import xlrd
import csv
import openpyxl
from sklearn.model_selection import train_test_split
import tensorflow as tf
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
from collections import Counter
import keras
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation,Embedding,SpatialDropout1D,Conv1D,Flatten,MaxPooling1D
from preprocessing import full_pre_process
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
<<<<<<< HEAD

max_words = 5000
max_len = 200

c,tweets,y=full_pre_process()
y = np.array(y)
labels = tf.keras.utils.to_categorical(y, 2, dtype="int32")
del y

embedding_layer = Embedding(300, 40)
X_train, X_test_val, y_train, y_test_val = train_test_split(tweets,labels,test_size=0.3, random_state=0)

X_test, X_val, y_test, y_val= train_test_split( X_test_val, y_test_val,test_size=0.7, random_state=0)    
model1 = Sequential()
model1.add(Embedding(max_words, 30))
model1.add(LSTM(15,dropout=0.5))
model1.add(Dense(2,activation='softmax'))


model1.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
#Implementing model checkpoins to save the best metric and do not lose it on training.
checkpoint1 = ModelCheckpoint("best_model1.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
history = model1.fit(X_train, y_train, epochs=70,validation_data=(X_val, y_val),callbacks=[checkpoint1])

predictions = model1.predict(X_test)


<<<<<<< HEAD
def LstmModel():
    
   
    cleanTweets,X,sentiment=full_pre_process()
    X_train,X_test_val,Y_train,Y_test_val=train_test_split(X,sentiment,test_size=0.1)
    X_val,X_test,Y_val,Y_test=train_test_split(X_test_val,Y_test_val)
    Y_train=np_utils.to_categorical(Y_train)
    Y_test=np_utils.to_categorical(Y_test)
    Y_val=np_utils.to_categorical(Y_val)
    num_classes=Y_test.shape[1]
    #build model
    
    model = Sequential()
    forward_layer = LSTM(6, return_sequences=True, input_shape=(40,300))
    backward_layer = LSTM(6, activation='relu', return_sequences=True,go_backwards=True)
    model.add(Bidirectional(forward_layer, backward_layer=backward_layer))
    model.add(Dropout(0.2))
    model.add(Dense(30,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(2,activation="softmax"))
    history=model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    print(history)
    #Implementing model checkpoins to save the best metric and do not lose it on training.
    checkpoint1 = ModelCheckpoint("best_model1.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
    model.fit(X_train, Y_train, epochs=50,validation_data=(X_val, Y_val),callbacks=[checkpoint1])
    predictions = model.predict(X_val)
  
    return  predictions,Y_val
predictions,labels=LstmModel()
=======
>>>>>>> parent of 9e682a8... Revert "lstm works!!!"
=======

max_words = 5000
max_len = 200

c,tweets,y=full_pre_process()
y = np.array(y)
labels = tf.keras.utils.to_categorical(y, 2, dtype="int32")
del y

embedding_layer = Embedding(300, 40)
X_train, X_test_val, y_train, y_test_val = train_test_split(tweets,labels,test_size=0.3, random_state=0)

X_test, X_val, y_test, y_val= train_test_split( X_test_val, y_test_val,test_size=0.7, random_state=0)    
model1 = Sequential()
model1.add(Embedding(max_words, 30))
model1.add(LSTM(15,dropout=0.5))
model1.add(Dense(2,activation='softmax'))


model1.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
#Implementing model checkpoins to save the best metric and do not lose it on training.
checkpoint1 = ModelCheckpoint("best_model1.hdf5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', period=1,save_weights_only=False)
history = model1.fit(X_train, y_train, epochs=70,validation_data=(X_val, y_val),callbacks=[checkpoint1])

predictions = model1.predict(X_test)


>>>>>>> parent of 9e682a8... Revert "lstm works!!!"
