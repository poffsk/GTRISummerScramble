#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import foolbox
import eagerpy as ep
import foolbox


mnist=tf.keras.datasets.mnist
from sklearn.model_selection import KFold

kf=KFold(n_splits=5)

kf.get_n_splits()

(x_train, y_train),(x_test,y_test)=mnist.load_data()

x_train,x_test=x_train/255.0,x_test/255.0


model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions=model(x_train[:5]).numpy()
#predictions

tf.nn.softmax(predictions).numpy()

loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:5],predictions).numpy()

model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])


for train_index, test_index in kf.split(x_train):
    x_trains, x_tests=x_train[train_index],x_train[test_index]
    y_trains, y_tests=y_train[train_index], y_train[test_index]
    
    model.fit(x_trains,y_trains, validation_data=(x_tests,y_tests), epochs=5)
    

predict=model(x_test[:1]).numpy()
tf.nn.softmax(predict).numpy()


correct_pred=tf.math.argmax(predict,1)
# print(x_test[:1])
# print(correct_pred)
# print(y_test[:1])

model.evaluate(x_test,y_test,verbose=2)


# attacks= [
# 	foolbox.attacks.FGSM(),
# 	foolbox.attacks.LinfPGD(),
# ]

# epsilons=[
# 	0.0,
# 	0.001,
# 	0.01,
# 	0.03,
# 	0.1,
# 	0.3,
# 	0.5,
# 	1.0,
# ]

#print(x_test[0],y_test[0])
foolbox_model=foolbox.models.TensorFlowModel(
	model=model,
	bounds=(0,1),
)
foolbox_model=foolbox_model.transform_bounds((0,1))
assert foolbox_model.bounds==(0,1)

images=ep.astensors(x_test[0:16])
labels=ep.astensors(y_test[0:16])

#doesn't work right
attack=foolbox.attacks.LinfPGD()
_, advs, success =attack(foolbox_model, images, labels, epsilons=0.03)

print(success)


# The end accuracy was 0.9809 compared to 0.9496 when not using KFold.
# 
# The predictions for the first 5:
# array([[5.43997092e-23, 4.90765706e-14, 2.98600290e-15, 6.97289174e-03,
#         4.53084188e-32, 9.93027031e-01, 6.42024900e-24, 4.53798596e-17,
#         2.48018923e-16, 5.83970931e-16],
#        [9.99999881e-01, 1.24962863e-16, 1.35353190e-07, 9.39377114e-16,
#         3.49472651e-22, 9.76741220e-17, 1.71943284e-14, 1.57634526e-14,
#         1.21071154e-14, 6.09650941e-11],
#        [1.46871773e-19, 4.17203294e-12, 2.86911583e-09, 1.69426088e-11,
#         1.00000000e+00, 1.73421832e-10, 1.14317405e-14, 1.98199923e-09,
#         5.04708211e-12, 1.27709097e-08],
#        [5.47637421e-15, 1.00000000e+00, 3.38236017e-09, 1.02122017e-11,
#         3.74660136e-10, 2.56695029e-15, 1.02424605e-14, 5.23055732e-08,
#         1.32038496e-08, 5.34089842e-19],
#        [1.06777146e-19, 1.38174722e-10, 1.98083164e-11, 5.10287208e-08,
#         6.39576228e-06, 3.26672370e-11, 1.66266695e-17, 9.62504298e-09,
#         4.53664946e-08, 9.99993443e-01]], dtype=float32)

# In[48]:
# In[49]:
#predict
# # In[50]:
# results=model.predict(x_test)
# # In[51]:
# results=np.argmax(results,axis=1)
# # In[52]:
# results=pd.Series(results,name="Label")
# print(results)
# # In[53]:
# submission=pd.concat([pd.Series(range(1,10001),name="ImageId"),results],axis=1)
# # In[54]:
# submission.to_csv("cnnmnist.csv",index=False)
# In[ ]:

# Use Foolbox to create some adversarial examples given the MNIST data set and your MNIST model. 
# For starters, study the minimal example given in: https://github.com/bethgelab/foolbox#example

# Try different attacks (in the API docs) like Fast Gradient Sign Method (FGSM) or the Projected Gradient Descent (PGD). 
# Visualize the before and after attack image. What was the original digit, and what does the classifier think it is now?
# Pick one additional attack to generate adversarial examples, 
# like Boundary Attack or DeepFool.
# Pick 100 examples from the MNIST dataset to run 3 attacks 
# (FGSM, PGD, one of your choice) and compare:
# how many of those 100 did the attack successfully find
# adversarial examples?
# What was the original class, and what class did it get changed to?
# What is the average L2 norm of the original vs perturbed samples
# (ie what is the measure of perturbation)? You may use scikit-learn’s metric module to help with calculations
# What’s the average time each attack took to find an adversarial example?
# Which attack would you recommend?
# Put the answers to subquestions 1, 3, 4 in a results table. If you want, use Matplotlib to make a bar chart with the answers.
# Put results and chart in your GitHub page week 2 folder.




