# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 18:28:12 2017

@author: Spandan_Mishra
"""
import pandas as pd
from pandas import ExcelWriter
import numpy as np
from keras.layers import  Conv2D, MaxPool2D, BatchNormalization, Dense, Dropout, Flatten, Activation
from keras.models import Sequential

#from keras.layers import to_categorical 

def get_images(df):
    '''Create 3-channel 'images'. Return rescale-normalised images.'''
    images = []
    for i, row in df.iterrows():
        # Formulate the bands as 75x75 arrays
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        # Rescale
        r = (band_1 - band_1.min()) / (band_1.max() - band_1.min())
        g = (band_2 - band_2.min()) / (band_2.max() - band_2.min())
        b = (band_3 - band_3.min()) / (band_3.max() - band_3.min())

        rgb = np.dstack((r, g, b))
        images.append(rgb)
    return np.array(images)

def random_sampling(Xtrain, Ytrain, batchsize,seed):
    np.random(seed)
    m= Xtrain.shape[1]
    random_perm = np.random.permutation(range(m))
    complete_batch = np.floor(m/batchsize)
    X_random = Xtrain[:random_perm]
    Y_random = Ytrain[:random_perm]
    minibatches=[]
    for i in range(complete_batch):
        batchX = X_random[:,(i*batchsize):(i+1)*batchsize]
        batchY = Y_random[:,(i*batchsize):(i+1)*batchsize]
        minibatches.append((batchX,batchY))
    if(m % batchsize==0):
        return minibatches
    else:
        batchX = X_random[:,complete_batch:m]
        batchY = Y_random[:,complete_batch:m]
        minibatches.append((batchX,batchY))
        return minibatches
    
def forward_propagation(input_shape):
    model = Sequential()
    ################################################
    model.add(Conv2D( 6 , kernel_size =(5,5), strides =(1,1), padding = "same",name= "conv1", input_shape= input_shape))
    model.add(BatchNormalization(axis=3, name="bn0"))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2), name="pool1"))
    model.add(Dropout(0.5))
    ##############################################
    model.add(Conv2D(16,kernel_size=(5,5), strides=(1,1), name="conv2"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2),name="pool2"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    ######################################################    
    model.add(BatchNormalization(axis=1,name="bn1"))
    model.add(Dense(100,activation="relu"))  
    model.add(Dropout(0.5))
    ##########################################
    model.add(Dense(24,activation="relu"))
    model.add(BatchNormalization(axis=1,name="bn2"))
    model.add(Dense(1,activation="sigmoid"))
    return model

train = pd.read_json('train.json')
#test = pd.read_json('test.json')
#Xtest= get_images(test)
Xtrain = get_images(train)
Xtrain = Xtrain / 255
Ytrain = np.array(train.is_iceberg)

model = forward_propagation(input_shape= Xtrain.shape[1:])
model.compile(loss ='binary_crossentropy', optimizer = "adam",metrics=["accuracy"])
model.fit(Xtrain, Ytrain, batch_size =64, epochs = 200, verbose=0, callbacks=[plot_losses],validation_split =0.1)
score, accuracy = model.evaluate(Xtrain,Ytrain,verbose =1)

#model.save('Glacier_LENET.h5')
############################
#test_predict= model.predict(Xtest)
#ID = np.array(test.id)
#df = pd.DataFrame({'id': ID, 'is_iceberg': np.squeeze(test_predict)})
#writer = pd.ExcelWriter("C:\\Users\\Spandan Mishra\\Documents\\GitHub\\IcebergChallenge\\TestPrediction.xlsx",engine='xlsxwriter')
#df.to_excel(writer, sheet_name="Sheet 1") 
#######################

	









