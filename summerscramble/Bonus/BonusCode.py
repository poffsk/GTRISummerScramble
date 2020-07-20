# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 18:24:58 2020

@author: Sarah
"""

"""This time we define a large CNN architecture with additional convolutional,
 max pooling layers and fully connected layers.
 The network topology can be summarized as follows.
 
 
Convolutional layer with 30 feature maps of size 5×5.
Pooling layer taking the max over 2*2 patches.
Convolutional layer with 15 feature maps of size 3×3.
Pooling layer taking the max over 2*2 patches.
Dropout layer with a probability of 20%.
Flatten layer.
Fully connected layer with 128 neurons and rectifier activation.
Fully connected layer with 50 neurons and rectifier activation.
Output layer."""

# Larger CNN for the MNIST Dataset
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
#import foolbox as fb
import matplotlib.pyplot as plt
import json
# from sklearn.model_selection import cross_val_score
# Final evaluation of the model
#clf = svm.SVC(kernel='linear', C=1)
#scores = cross_val_score(clf, iris.data, iris.target, cv=5)

#Extract data
with open('dict_train.json') as train_file:
    data = json.load(train_file)
    
with open('test_data.json') as test_file:
    testdata = json.load(test_file)
    

""">>> data.keys()
dict_keys(['data', 'row_index', 'column_index', 'schema', 'shape', 'labels'])"""

#15670 instances total
#3133 will be test, 12537 are examples

datalist = data['data'] #has length of 6317199, range of 1-56 (0 is default), I assume this is counts
rowlist = data['row_index'] #6317199, (0, 12535)
collist = data['column_index'] #6317199 (0, 106427)
col_schema = data['schema'] #106428, 1111 API feature and their combos (1, 2, or 3 at a time)
shape = data['shape'] #[12536, 106428] = #labels, #schema
labels = data['labels'] #12536, number of training examples

num_APIs = 0
for i in range(0, len(col_schema)):
    strnums = col_schema[i].split()
    for j in strnums:
        if int(j) > num_APIs:
            num_APIs = int(j)

#start with just unigrams
#make list of old column numbers and map them to new column numbers

maplist=[] #will have ordered pair (oldcolnumber, newcolnumber)
uni_col_schema = [] #just going to be ['0', '1', ..., '1110']
#want to go from 106428 columns to 1111 for unigram 
for i in range(0, len(col_schema)):
    if len(col_schema[i].split()) == 1: #there is just a unigram
        maplist.append((i, int(col_schema[i])))
        uni_col_schema.append(col_schema[i])
#col_schema[106423] = '1110'
        
        
#Parse down datalist to keep only unigrams
uni_datalist = [] #will be abridged
uni_rowlist = [] #will be abridged
uni_collist = [] #will be mapped from maplist
#labels will be the same bc we aren't deleting examples, just abridging info

for i in range(0, len(datalist)):
    for j in range(0, len(maplist)): #is there a way to avoid this second loop? maybe I shouldn't map
        if collist[i] == maplist[j][0]: #we found a match to keep
            uni_datalist.append(datalist[i]) #keep that datapoint
            uni_rowlist.append(rowlist[i]) #keep that row index
            uni_collist.append(maplist[j][1]) #add the new, abridged column

feature_matrix = np.zeros((12536, 1111))
for pos, value in enumerate(uni_datalist):
    feature_matrix[uni_rowlist[pos], uni_collist[pos]] = value

training_labels = np.array(labels)
                           
                           

#let's save this to a json for tomorrow
uni_dict={}
uni_dict['uni_datalist'] = uni_datalist
uni_dict['uni_rowlist'] = uni_rowlist
uni_dict['uni_collist'] = uni_collist
uni_dict['maplist'] = maplist
uni_dict['uni_col_schema'] = uni_col_schema
uni_dict['labels'] = labels

with open('unidata.json', 'w', encoding='utf-8') as f:
    json.dump(uni_dict, f, ensure_ascii=False, indent=4)

# load data
X_train = feature_matrix
y_train = training_labels
#(X_train, y_train), (X_test, y_test) = feature_matrix, test_matrix
# reshape to be [samples][width][height][channels]
#X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
#X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / (num_APIs + 1) #may need to also count test data APIs
#X_test = X_test / (num_APIs + 1) #bc don't want to give value greater than one
#X_train = X_train / 255
#X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
#y_test = np_utils.to_categorical(y_test)
#num_classes = y_test.shape[1]
kf = KFold(n_splits=5)
kf.get_n_splits(X_train)

""">>> kf.get_n_splits(X)
2
>>> print(kf)
KFold(n_splits=2, random_state=None, shuffle=False)"""

#let's start with just dense layer connecting 1111 inputs to 5 outputs
def larger_model():
    model = Sequential()
    model.add(Flatten(input_shape = (1111,)))
    model.add(Dropout(0.2))
    model.add(Dense(75, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.2), loss='categorical_crossentropy', metrics=['accuracy'])
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    """model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu')) #modify
    model.add(MaxPooling2D()) #modify
    model.add(Conv2D(15, (3, 3), activation='relu')) #modify
    model.add(MaxPooling2D()) #modify?
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax')) """
    
# build the model
model = larger_model()

scoresvec = []

for train_index, test_index in kf.split(X_train): #might need to tack on targets here
    print("TRAIN:", train_index, "TEST:", test_index)
    X_trainK, X_testK = X_train[train_index], X_train[test_index]
    y_trainK, y_testK = y_train[train_index], y_train[test_index]
    model.fit(X_trainK, y_trainK, validation_data=(X_testK, y_testK), epochs=5, batch_size=200)
  #  scores = model.evaluate(X_test, y_test, verbose=0) #should I change this verbose value?
  #  scoresvec.append((100-scores[1]*100))

#CUDA, total of three
    #cuda 10.1...
#y_pred = model.predict(X_test[0:101])
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)