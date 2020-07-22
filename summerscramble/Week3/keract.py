# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:48:09 2020

@author: Sarah
"""

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
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from sklearn.model_selection import KFold
#import eagerpy as ep
import matplotlib.pyplot as plt
import json
import csv
#import foolbox as fb
#from foolbox import TensorFlowModel, accuracy, samples
#from foolbox.attacks import LinfPGD
#from foolbox.attacks import FGSM
# from sklearn.model_selection import cross_val_score
# Final evaluation of the model
#clf = svm.SVC(kernel='linear', C=1)
#scores = cross_val_score(clf, iris.data, iris.target, cv=5)

"""
#extra adversarial example images from Camille
image_pre_array=[]

with open('dfadvpics.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_pre_array.append(row)

#start with 16 rows, 784 columns       
image_array=np.zeros(16, 28, 28, 1)
image_keys = ['image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'image8', 'image9', 'image10', 'image11', 'image12', 'image13', 'image14', 'image15', 'image16']
i=-1
j=-1

#keys are '', image1, ..., image16
for row in image_pre_array: #16 or 17 entries per row, 784 rows
    i+=1 #row number
    for key in image_keys:
        j+=1 #value number
    image_array[int(key[-1])-1][i][j]= int(row[key])"""

#configure model
    
img_width, img_height = 28, 28
batch_size = 200
no_epochs = 5
no_classes = 10
validation_split = 0.2
verbosity = 0


kf = KFold(n_splits=5)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][width][height][channels]


#reshape data based on if channels ordering (first or last):
if K.image_data_format()=='channels_first':
    X_train = X_train.reshape((X_train.shape[0], 1, img_width, img_height)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], 1, img_width, img_height)).astype('float32')
    input_shape = (1, img_width, img_height)
else:
    X_train = X_train.reshape((X_train.shape[0], img_width, img_height, 1)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], img_width, img_height, 1)).astype('float32')
    input_shape = (img_width, img_height, 1)
    

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
#y_train = tf.keras.utils.to_categorical(y_train, no_classes)
#y_test = tf.keras.utils.to_categorical(y_test, no_classes)
#num_classes = y_test.shape[1]
""">>> kf.get_n_splits(X_train)
2
>>> print(kf)
KFold(n_splits=2, random_state=None, shuffle=False)"""


def larger_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #model.add(Dense(50, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build the model
model = larger_model()

"""# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))"""

scoresvec = []

for train_index, test_index in kf.split(X_train): #might need to tack on targets here
    print("TRAIN:", train_index, "TEST:", test_index)
    X_trainK, X_testK = X_train[train_index], X_train[test_index]
    y_trainK, y_testK = y_train[train_index], y_train[test_index]
    model.fit(X_trainK, y_trainK, validation_data=(X_testK, y_testK), epochs=5, batch_size=100)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {scores[0]} / Test accuracy: {scores[1]}')
    scoresvec.append((100-scores[1]*100))

#CUDA, total of three
    #cuda 10.1...
y_pred = model.predict(X_test[0:101])


from keract import get_activations, display_activations
#keract_inputs = X_train[:1]
#keract_target= y_train[:1]
#actives=get_activations(model, keract_inputs, layer_names = 'dense')
#X_train = X_train.reshape((X_train.shape[0], img_width, img_height, 1)).astype('float32')

active_sum=np.zeros((no_classes, 128)) #each row is a class, 128 cols = sum of each node
active_sum2=np.zeros((no_classes, 128))
class_obs=np.zeros((10))
for i in range(0, len(X_train)):
    keract_inputs = X_train[i:i+1]
    keract_targets = y_train[i]
    img_class = int(np.where(keract_targets == 1)[0][0])
    actives=get_activations(model, keract_inputs, layer_names = 'dense')
    layer_vals = actives['dense'][0]
    active_sum[img_class] = active_sum[img_class] + layer_vals
    active_sum2[img_class] = active_sum2[img_class] + np.square(layer_vals)
    #we will also need to know how many of each class we observed
    class_obs[img_class] +=1
    
#display_activations(actives, cmap='gray', save=False)

#heatmaps
from keract import display_heatmaps
display_heatmaps(actives, keract_inputs, save=False)
#model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)


#images = X_test[0:101]
#labels = y_test[0:101]


#fmodel = create()
#fmodel = fb.PyTorchModel(model=model, bounds=(0, 1))
#return fmodel
# https://foolbox.readthedocs.io/en/v2.3.0/user/examples.html
#fmodel = fb.models.TensorFlowModel(model=model, bounds = (0.0,1.0), device = None) #this one crashes
#fmodel = TensorFlowModel(model, bounds = (0,1))
#nooo it crashed

#images, labels = samples(fmodel, dataset = "mnist", batchsize = 100)
#images1, labels1 = ep.astensors(*samples(fmodel, dataset = "mnist", batchsize = 100)) 
#print(accuracy(fmodel, images1, labels1))


images_arr = np.array(images)
#plot images before attack
fig, axes = plt.subplots(1, 100, figsize = (28,28))
axes = axes.flatten()
for img, ax in zip(images_arr, axes):
    ax.imshow(np.squeeze(img), cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()


epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]


fmodel = foolbox.models.TensorFlowModel(model, bounds = (0.0,1.0)) #this one crashes
fmodel = fmodel.transform_bounds((0,1))
#assert fmodel.bounds==(0,1)

#images = X_test[0:101]
#labels = y_test[0:101]

#images, labels = ep.astensors(*samples(fmodel, dataset="mnist", batchsize=20))
images, labels = ep.astensors(*samples(images, labels))
#print(accuracy(fmodel, images, labels))


attack = fb.attacks.LinfPGD()
_, advs, success = attack(fmodel, images, labels, epsilons= 0.03)

print(success)


listsuccess = []
for anepval in success:
    print(anepval)
    totalsuccess=0
    for abool in anepval:
        abool1 = np.array(abool)
        if abool1==True:
            totalsuccess+=1
    listsuccess.append(totalsuccess)
    
print(listsucess)


#with tf.keras.backend.get_session().as_default():
#    foolbox_model = foolbox.models.TensorFlowModel.from_keras(model=network, bounds=(0.0, 1.0))

# https://foolbox.readthedocs.io/en/v2.3.0/user/examples.html



predict = fmodel(images).numpy()
#I wonder what the two lines below do
tf.nn.softmax(predict).numpy()
correct_pred = tf.math.argmax(predict, 1)
print(correct_pred)



