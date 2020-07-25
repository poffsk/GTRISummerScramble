#visualize layer activations of a tensorflow.keras CNN with Keract (tutorial)

import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras import activations

#configure model
img_width, img_height =28, 28
batch_size=250
no_epochs=25
no_classes=10
validation_split=0.2
verbosity=1

#Load mnist dataset
(input_train, target_train),(input_test, target_test)=mnist.load_data()

#reshape data based on channels first/channels last

if K.image_data_format()=='channels_first':
	input_train=input_train.reshape(input_train.shape[0], 1, img_width, img_height)
	input_test=input_test.reshape(input_test.shape[0], 1, img_width, img_height)
	input_shape=(1, img_width, img_height)
else:
	input_train=input_train.reshape(input_train.shape[0], img_width, img_height, 1)
	input_test=input_test.reshape(input_test.shape[0], img_width, img_height, 1)
	input_shape=(img_width, img_height, 1)

#parse numbers as floats
input_train=input_train.astype('float32')
input_test=input_test.astype('float32')

#normalize the data
input_train=input_train/255
input_test=input_test/255

#convert target vectors to categorical targets
target_train=tf.keras.utils.to_categorical(target_train, no_classes)
target_test=tf.keras.utils.to_categorical(target_test, no_classes)


#create the model
model=Sequential()
model.add(Conv2D(6, kernel_size=(5,5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(10, kernel_size=(5, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

#compile the model
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

#fit data to model
model.fit(input_train, target_train, batch_size=batch_size, epochs=no_epochs, verbose=verbosity, validation_split=validation_split)

#generate generalization metrics
score=model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')



#keract visualizations
#for each layer
from keract import get_activations, display_activations
keract_inputs=input_test[:1]
keract_target=target_test[:1]
activations=get_activations(model, keract_inputs)
display_activations(activations, cmap='gray', save=False)


#heatmaps
from keract import display_heatmaps
display_heatmaps(activations, keract_inputs, save=False)

