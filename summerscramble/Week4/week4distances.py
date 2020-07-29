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
from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from sklearn.model_selection import KFold
#import eagerpy as ep
import matplotlib.pyplot as plt
import json
import csv
from keract import get_activations, display_activations
from numpy import genfromtxt
from keract import display_heatmaps
import matplotlib.pyplot as plt
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
"""if K.image_data_format()=='channels_first':
    X_train = X_train.reshape((X_train.shape[0], 1, img_width, img_height)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], 1, img_width, img_height)).astype('float32')
    input_shape = (1, img_width, img_height)
else:
    X_train = X_train.reshape((X_train.shape[0], img_width, img_height, 1)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], img_width, img_height, 1)).astype('float32')
    input_shape = (img_width, img_height, 1)
"""

"""def larger_model():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
#    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
#    model.add(Dense(50, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# build the model
model = larger_model()

from tensorflow import keras
model = keras.models.load_model('model')"""

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

"""scoresvec = []

for train_index, test_index in kf.split(X_train): #might need to tack on targets here
    X_trainK, X_testK = X_train[train_index], X_train[test_index]
    y_trainK, y_testK = y_train[train_index], y_train[test_index]
    model.fit(X_trainK, y_trainK, validation_data=(X_testK, y_testK), epochs=5, batch_size=200)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {scores[0]} / Test accuracy: {scores[1]}')
    scoresvec.append((100-scores[1]*100))"""


from tensorflow import keras
model = keras.models.load_model('model')
y_pred = model.predict(X_test[0:101])

active_sum=np.zeros((no_classes, 128)) #each row is a class, 128 cols = sum of each node
active_sum2=np.zeros((no_classes, 128))
class_obs=np.zeros((10))

"""
stored_layers=np.zeros((100, 128))
for i in range(0, 5001):
    keract_inputs = X_train[i:i+1]
    keract_targets = y_train[i]
    img_class = int(np.where(keract_targets == 1)[0][0])
    actives=get_activations(model, keract_inputs, layer_names = 'dense')
    layer_vals = actives['dense'][0]
    stored_layers[i] = layer_vals
    #active_sum[img_class] = active_sum[img_class] + layer_vals
    #active_sum2[img_class] = active_sum2[img_class] + np.square(layer_vals)
    #we will also need to know how many of each class we observed
    class_obs[img_class] += 1

#let's see how normal this thing is...
class_obs=np.zeros((10))
stored_layers=np.zeros((1000, 128))
for i in range(0,len(b1000_indices)):
    for j in range(0, len(b1000_indices[i])):
        index = int(b1000_indices[i][j])
        keract_inputs = X_train[index:index+1]
        keract_targets = y_train[index]
        img_class = int(np.where(keract_targets == 1)[0][0])
        actives=get_activations(model, keract_inputs, layer_names = 'dense')
        layer_vals = actives['dense'][0]
        stored_layers[i*100+j] = layer_vals
    #active_sum[img_class] = active_sum[img_class] + layer_vals
    #active_sum2[img_class] = active_sum2[img_class] + np.square(layer_vals)
    #we will also need to know how many of each class we observed
        class_obs[img_class] += 1

"""

#------Plotting Neuron Activations By Class------------
neuron_one = []
for i in range(0,1000):
    neuron_one.append(stored_layers[i][1])
    
#plt.hist(neuron_one[0:10], density=False, bins=5)  # `density=False` would make counts
#plt.ylabel('Frequency')
#plt.xlabel('Value');
    
i=0
plt.hist(neuron_one[100*i:100*i+99], density=False, bins=10)
plt.hist(hist_data, density=False, bins=15)

      
fig, axs = plt.subplots(3,3)
i=1
axs[0,0].hist(neuron_one[100*i:100*i+99], density=False, bins=10)
i=2
axs[0,1].hist(neuron_one[100*i:100*i+99], density=False, bins=10)
i=3
axs[0,2].hist(neuron_one[100*i:100*i+99], density=False, bins=10)
i=4
axs[1,0].hist(neuron_one[100*i:100*i+99], density=False, bins=10)
i=5
axs[1,1].hist(neuron_one[100*i:100*i+99], density=False, bins=10)
i=6
axs[1,2].hist(neuron_one[100*i:100*i+99], density=False, bins=10)
i=7
axs[2,0].hist(neuron_one[100*i:100*i+99], density=False, bins=10)
i=8
axs[2,1].hist(neuron_one[100*i:100*i+99], density=False, bins=10)
i=9
axs[2,2].hist(neuron_one[100*i:100*i+99], density=False, bins=10)


#---------------End Plotting--------------------
"""
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
    """
    
#with open(r'C:\Users\Sarah\Desktop\Su20\DNN\summerscramble\summerscramble\Week3\keract_actives.json', 'w', encoding='utf-8') as f:
#    json.dump(activation_dict, f, ensure_ascii=False, indent=4)
            
            
#Let's try to save some of these arrays...
np.savetxt("active_mean.csv", active_mean, delimiter=",")
np.savetxt("active_sum.csv", active_sum, delimiter=",")
np.savetxt("active_sum2.csv", active_sum2, delimiter=",")
np.savetxt("class_obs.csv", class_obs, delimiter=",")
np.savetxt("active_stdev.csv", active_stdev, delimiter=",")
np.savetxt("beyond_one_stdev.csv", beyond_one_stdev, delimiter=",")
np.savetxt("stored_layers.csv", stored_layers, delimiter= ",")

active_mean = genfromtxt('active_mean.csv', delimiter=',')
active_sum = genfromtxt('active_sum.csv', delimiter=',')
active_sum2 = genfromtxt('active_sum2.csv', delimiter=',')
class_obs = genfromtxt('class_obs.csv', delimiter=',')
active_stdev = genfromtxt('active_stdev.csv', delimiter=',')
beyond_one_stdev = genfromtxt('beyond_one_stdev.csv', delimiter=',')
stored_layers = genfromtxt('stored_layers.csv', delimiter=',')


#-------Adversarial Section----------------
#create arrays of adversarial images from Camille
#create array of 100 images to run adversarial attack on
"""benign_100=np.zeros((100, 28, 28, 1))
jarray = np.zeros(10)
b100_indices = np.zeros((10,10))
while np.sum(jarray) < 100:
    for i in range(0,len(X_train)):
        keract_inputs = X_train[i:i+1]
        keract_targets = y_train[i]
        img_class = int(np.where(keract_targets == 1)[0][0])
        if jarray[img_class] < 10: #if we don't already have 10
            count = int(jarray[img_class])
            b100_indices[img_class][count] = i
            jarray[img_class]+=1

for i in range(0, 10):
    for j in range(0, 10):
        index = int(b100_indices[i][j])
        benign_100[i*10 + j] = X_train[index:index+1]
"""

dfadv1pics=genfromtxt('dfadvpics.csv', delimiter=',')
dfadvpics2=genfromtxt('dfadvspics2.csv', delimiter=',')
dfadv_by_class = genfromtxt('dfadv_by_class.csv', delimiter=',')

adv1pics = np.transpose(dfadv1pics)
adv1pics = adv1pics[1:, 1:]
image1_array=np.zeros((len(adv1pics), 28, 28, 1))
for i in range(0, len(adv1pics)):
    image1_array[i] = adv1pics[i].reshape(28, 28, 1)


adv2pics = np.transpose(dfadvpics2)
adv2pics = adv2pics[1:, 1:]
image2_array=np.zeros((len(adv2pics), 28, 28, 1))
for i in range(0, len(adv2pics)):
    image2_array[i] = adv2pics[i].reshape(28, 28, 1)
 
#all 100 images
advpics = np.transpose(dfadv_by_class)
advpics = advpics[1:, 1:]
image_array=np.zeros((len(advpics), 28, 28, 1))
for i in range(0, len(advpics)):
    image_array[i] = advpics[i].reshape(28, 28, 1)
   
adv_active_sum=np.zeros((no_classes, 128)) #each row is a class, 128 cols = sum of each node
adv_active_sum2=np.zeros((no_classes, 128))
adv_class_obs=np.zeros((no_classes))
for i in range(0, len(image_array)):
    keract_inputs = image_array[i:i+1]
    keract_targets = y_train[i] #assuming they were produced in same order
    img_class = int(np.where(keract_targets == 1)[0][0])
    actives=get_activations(model, keract_inputs, layer_names = 'dense')
    layer_vals = actives['dense'][0]
    adv_active_sum[img_class] = adv_active_sum[img_class] + layer_vals
    adv_active_sum2[img_class] = adv_active_sum2[img_class] + np.square(layer_vals)
    #we will also need to know how many of each class we observed
    adv_class_obs[img_class] += 1

"""adv_preds=np.zeros((100))
y_pred = model.predict(image_array)
for i in range(0, len(y_pred)): 
    img_class = int(np.where(y_pred[i] >= max(y_pred[i]))[0][0])
    adv_preds[i] = img_class

adv_preds.reshape(10,10)"""

#the first ten images were 0, then next 10 were one, etc to indexes 90-99
beyond_one_stdev_adv = np.zeros((no_classes, 128))
for i in range(0, 99):
    keract_inputs = image_array[i:i+1]
    #keract_targets = y_train[i] #what does this line do...
    img_class = int(round(i/10, 0))
    actives=get_activations(model, keract_inputs, layer_names = 'dense')
    layer_vals = actives['dense'][0]
    for j in range(0, len(layer_vals)):
        #if the difference between current activation and mean activation is larger than stdev
        if abs(layer_vals[j]-active_mean[img_class][j]) > active_stdev[img_class][j]:
            beyond_one_stdev_adv[img_class][j] +=1

#2853 are out of bounds according to their original class
np.savetxt("beyond_one_stdev_adv.csv", beyond_one_stdev_adv, delimiter=",")
beyond_one_stdev_adv = genfromtxt('beyond_one_stdev_adv.csv', delimiter=',')

#this one will compare the 100 images to their adversarial classes
y_pred = model.predict(image_array)

beyond_one_stdev_adv2 = np.zeros((no_classes, 128))
for i in range(0, 99):
    keract_inputs = image_array[i:i+1]
    keract_targets = y_pred[i]
    max_value = max(y_pred[i])
    img_class = int(np.where(keract_targets >= max_value)[0][0])
    actives=get_activations(model, keract_inputs, layer_names = 'dense')
    layer_vals = actives['dense'][0]
    for j in range(0, len(layer_vals)):
        #if the difference between current activation and mean activation is larger than stdev
        if abs(layer_vals[j]-active_mean[img_class][j]) > active_stdev[img_class][j]:
            beyond_one_stdev_adv2[img_class][j] +=1
    
#1813 are out of bounds according to their 'new' classes...
#I'd like to know how many are the same...
np.savetxt("beyond_one_stdev_adv2.csv", beyond_one_stdev_adv2, delimiter=",")
beyond_one_stdev_adv2 = genfromtxt('beyond_one_stdev_adv.csv2', delimiter=',')

for i in range(0,99):
    max_value = max(y_pred[i])
    img_class = int(np.where(y_pred[i] >= max_value)[0][0])
    print(img_class)
#wait, we don't need to calculate the 
#one two classified as a 7, one 5 classified as a 3    
#display_activations(actives, cmap='gray', save=False)
#------- End Adversarial Section----------------
    
    
#heatmaps
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


#-----------------showing heat map of activations--------------
import matplotlib
import matplotlib.pyplot as plt
image = X_train[0:1]
keract_inputs = image
keract_targets = y_train[0]
heat_array = np.zeros((1,16))
img_class = int(np.where(keract_targets == 1)[0][0])
actives=get_activations(model, keract_inputs, layer_names = 'dense')
layer_vals = actives['dense'][0][0:16]
for j in range(0, 16): #need to know num stdevs beyond, let's start with first 16 neurons
    if active_stdev[img_class][j] == 0:
        heat_array[0][j] = 0
    else:
        heat_array[0][j] = (layer_vals[j] - active_mean[img_class][j])/(active_stdev[img_class][j])

classes = [str(img_class)]
neurons = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16"]


#farmers are neurons, vegetables are classes
#need array where each row is a class and each column is a neuron's st devs above or below
fig, ax = plt.subplots()
im = ax.imshow(heat_array)

# We want to show all ticks...
ax.set_xticks(np.arange(len(neurons)))
ax.set_yticks(np.arange(len(classes)))
# ... and label them with the respective list entries
ax.set_xticklabels(neurons)
ax.set_yticklabels(classes)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(classes)):
    for j in range(len(neurons)):
        text = ax.text(j, i, round(heat_array[i, j],1),
                       ha="center", va="center", color="w")

ax.set_title("Standard deviations beyond observe for neurons of image")
fig.tight_layout()
plt.show()
    

#-----------------heat maps of activations complete--------

#Let's see which neurons we *really* need to consider...
from sklearn.decomposition import PCA
from scipy.stats import chi2

#image = images_array[0:1] #a five that the model thinks is a 3
#keract_inputs = image

five_images = np.zeros((len(five_indexes), 28, 28, 1))
for i in range(0, len(five_indexes)):
    five_images[i] = X_train[five_indexes[i]]
    
#five_indexes=[]
stored_5_layers=np.zeros((len(five_indexes)+1, 128))
for i in range(0, len(five_images)):
    keract_inputs = five_images[i:i+1]
#    keract_targets = y_train[i]
    actives=get_activations(model, keract_inputs, layer_names = 'dense')
    layer_vals = actives['dense'][0]
    stored_5_layers[i] = layer_vals
#    img_class = int(np.where(keract_targets == 1)[0][0])
#    if int(img_class) == 5:
#        five_indexes.append(i)  
    
#adversarial activations for first image
actives=get_activations(model, image_array[0:1], layer_names = 'dense')
adv_layer_vals = actives['dense'][0]

stored_5_layers[434] = adv_layer_vals

#we have 434 examples from the 5 class
pca = PCA(n_components = 35)
proj=pca.fit_transform(stored_5_layers)
#adv_transform=pca.transform(adv_layer_vals, 35)
var_5_ratios = pca.explained_variance_ratio_
print(pca.explained_variance_ratio_)


#Here is the good stuff
#x=pca.transform(stored_5_layers)
x=proj
x_mean=np.zeros((35))
for i in range(0, len(x)):
    x_mean = x_mean + x[i]
    
x_mean = x_mean / len(proj)
adv_x = proj[434] #434 is the adversarial
x_minus_mean = adv_x - x_mean
cov = np.cov(proj.T)
inv_covmat  = np.linalg.inv(cov)
left_term = np.dot(x_minus_mean, inv_covmat)
mahal = np.dot(left_term, x_minus_mean.T)
#mahala = 224.68914, so now we need a p value
#so transform the stored layouts
#and also transform that one image... how do we make sure same transformation?
#then calculate mahalanobis distance from image to class [means...]

#now we calculate p value

p_val = 1 - chi2.cdf(mahal, 34)

#p_val = 0 for adversarial, p_val = .202 for normal