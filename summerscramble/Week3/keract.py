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


#configure model
    
img_width, img_height = 28, 28
batch_size = 200
no_epochs = 5
no_classes = 10
validation_split = 0.2
verbosity = 0


kf = KFold(n_splits=5)

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
X_train = X_train.reshape((X_train.shape[0], img_width, img_height, 1)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], img_width, img_height, 1)).astype('float32')
input_shape = (28, 28, 1)
   

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train, no_classes)
y_test = np_utils.to_categorical(y_test, no_classes)
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
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_trainK, X_testK = X_train[train_index], X_train[test_index]
    y_trainK, y_testK = y_train[train_index], y_train[test_index]
    model.fit(X_trainK, y_trainK, validation_data=(X_testK, y_testK), epochs=5, batch_size=100)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {scores[0]} / Test accuracy: {scores[1]}')
    scoresvec.append((100-scores[1]*100))


y_pred = model.predict(X_test[0:101])


from keract import get_activations, display_activations


active_sum=np.zeros((no_classes, 128)) #each row is a class, 128 cols = sum of each node
active_sum2=np.zeros((no_classes, 128))
class_obs=np.zeros((10))
for i in range(0, 5001):
    keract_inputs = X_train[i:i+1]
    keract_targets = y_train[i]
    img_class = int(np.where(keract_targets == 1)[0][0])
    actives=get_activations(model, keract_inputs, layer_names = 'dense')
    layer_vals = actives['dense'][0]
    active_sum[img_class] = active_sum[img_class] + layer_vals
    active_sum2[img_class] = active_sum2[img_class] + np.square(layer_vals)
    #we will also need to know how many of each class we observed
    class_obs[img_class] += 1

active_mean=np.zeros((no_classes, 128))
divided_squares=np.zeros((no_classes, 128))
#squared_sums=np.zeros((no_classes, 128))
for i in range(0, len(active_sum)):
    active_mean[i] = active_sum[i]/ class_obs[i]
    divided_squares[i] = active_sum2[i] / class_obs[i]
    #squared_sums = np.square(active_sum[i] / class_obs[i])
#so now that we have the means and standard deviations... let's create boundaries

active_stdev=np.zeros((no_classes, 128))
for i in range(0, len(active_stdev)):
    class_var=np.subtract(divided_squares[i], np.square(active_mean[i]))
    active_stdev[i] = np.sqrt(class_var)

#now that we have active_mean and active_stdev, we need to go back through the examples and tally the number beyonds one stdev
beyond_one_stdev = np.zeros((no_classes, 128))
for i in range(0, 5001):
    keract_inputs = X_train[i:i+1]
    keract_targets = y_train[i]
    img_class = int(np.where(keract_targets == 1)[0][0])
    actives=get_activations(model, keract_inputs, layer_names = 'dense')
    layer_vals = actives['dense'][0]
    for j in range(0, len(layer_vals)):
        #if the difference between current activation and mean activation is larger than stdev
        if abs(layer_vals[j]-active_mean[img_class][j]) > active_stdev[img_class][j]:
            beyond_one_stdev[img_class][j] +=1
    
    
#with open(r'C:\Users\Sarah\Desktop\Su20\DNN\summerscramble\summerscramble\Week3\keract_actives.json', 'w', encoding='utf-8') as f:
#    json.dump(activation_dict, f, ensure_ascii=False, indent=4)
            
            
#Let's try to save some of these arrays...
np.savetxt("active_mean.csv", active_mean, delimiter=",")
np.savetxt("active_sum.csv", active_sum, delimiter=",")
np.savetxt("active_sum2.csv", active_sum2, delimiter=",")
np.savetxt("class_obs.csv", class_obs, delimiter=",")
np.savetxt("active_stdev.csv", active_stdev, delimiter=",")
np.savetxt("beyond_one_stdev.csv", beyond_one_stdev, delimiter=",")

from numpy import genfromtxt
active_mean = genfromtxt('active_mean.csv', delimiter=',')
active_sum = genfromtxt('active_sum.csv', delimiter=',')
active_sum2 = genfromtxt('active_sum2.csv', delimiter=',')
class_obs = genfromtxt('class_obs.csv', delimiter=',')
active_stdev = genfromtxt('active_stdev.csv', delimiter=',')
beyond_one_stdev = genfromtxt('beyond_one_stdev.csv', delimiter=',')


#create arrays of adversarial images from Camille
dfadvpics=genfromtxt('dfadvpics.csv', delimiter=',')
dfadvpics2=genfromtxt('dfadvspics2.csv', delimiter=',')

advpics = np.transpose(dfadvpics)
advpics = advpics[1:, 1:]
image_array=np.zeros((len(advpics), 28, 28, 1))
for i in range(0, len(advpics)):
    image_array[i] = advpics[i].reshape(28, 28, 1)


adv2pics = np.transpose(dfadvpics2)
adv2pics = adv2pics[1:, 1:]
image2_array=np.zeros((len(adv2pics), 28, 28, 1))
for i in range(0, len(adv2pics)):
    image2_array[i] = adv2pics[i].reshape(28, 28, 1)
    
    

adv_active_sum=np.zeros((no_classes, 128)) #each row is a class, 128 cols = sum of each node
adv_active_sum2=np.zeros((no_classes, 128))
adv_class_obs=np.zeros((no_classes))
for i in range(0, len(advpics)):
    keract_inputs = image_array[i:i+1]
    keract_targets = y_train[i] #assuming they were produced in same order
    img_class = int(np.where(keract_targets == 1)[0][0])
    actives=get_activations(model, keract_inputs, layer_names = 'dense')
    layer_vals = actives['dense'][0]
    adv_active_sum[img_class] = adv_active_sum[img_class] + layer_vals
    adv_active_sum2[img_class] = adv_active_sum2[img_class] + np.square(layer_vals)
    #we will also need to know how many of each class we observed
    adv_class_obs[img_class] += 1




    
#display_activations(actives, cmap='gray', save=False)

#heatmaps
from keract import display_heatmaps
display_heatmaps(actives, keract_inputs, save=False)



"""
images_arr = np.array(images)
#plot images before attack
fig, axes = plt.subplots(1, 100, figsize = (28,28))
axes = axes.flatten()
for img, ax in zip(images_arr, axes):
    ax.imshow(np.squeeze(img), cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()


epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]"""