import tensorflow as tf 
import eagerpy as ep 
import numpy as np 
from foolbox import TensorFlowModel, accuracy, samples
from foolbox.attacks import LinfPGD
from foolbox.attacks import FGSM
import matplotlib.pyplot as plt
#pre and advs, _, 


if __name__ == '__main__':

	mnist=tf.keras.datasets.mnist
	(x_train, y_train),(x_test,y_test)=mnist.load_data()
	x_train,x_test=x_train/255.0,x_test/255.0
	model=tf.keras.models.Sequential([
   		tf.keras.layers.Flatten(input_shape=(28,28)),
    	tf.keras.layers.Dense(128, activation='relu'),
   		tf.keras.layers.Dropout(0.2),
    	tf.keras.layers.Dense(10)
	])

	loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	model.compile(optimizer='adam',loss=loss_fn,metrics=['accuracy'])


	model.fit(x_train,y_train, epochs=5)

	model.evaluate(x_test,y_test,verbose=2)

	#instantiate the model
	fmodel=TensorFlowModel(model, bounds=(0,1))

	#get data and test the model
	#wrapping the tensors with ep.astensors is optional, but it allows
	#us to work with EagerPy tensors in the following

	##########################################################
	images, labels = samples(fmodel, dataset="mnist", batchsize=16)
	images1, labels1=ep.astensors(*samples(fmodel, dataset="mnist", batchsize=16))
	print(accuracy(fmodel, images1, labels1))


	predict=fmodel(images).numpy()
	tf.nn.softmax(predict).numpy()
	correct_pred=tf.math.argmax(predict,1)
	print(correct_pred)

	#print(images)
	images_arr=np.array(images)

	#print(images_arr)

	alist=[]
	alist=[0,1,2,3]
	#print(images_arr.shape) #16,28,28,1

	#not the good stuff...don't need
	# for i in range(len(alist)):
	# 	plt.subplots(1,4,figsize=(28,28))
	# 	plt.axis("off")
	# 	img_path=images_arr[i]
	# 	plt.imshow(np.squeeze(img_path))
	# 	plt.subplots_adjust(wspace=0.5)
	# plt.show()

	#plot images before attack... below is the good stuff

	fig, axes =plt.subplots(1,16,figsize=(28,28))
	axes=axes.flatten()
	for img, ax in zip(images_arr, axes):
		ax.imshow(np.squeeze(img), cmap="gray")
		ax.axis("off")
	plt.tight_layout()
	plt.show()

	###########################################################

	#apply the attack
	attack=LinfPGD()
	epsilons=[0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
	advs, _, success = attack(fmodel, images, labels, epsilons=epsilons)
	
	advs1=np.array(advs)
	advs1=advs1[3] #should be size 16, 28, 28, 1 

	predict2=fmodel(advs1).numpy()
	tf.nn.softmax(predict2).numpy()
	correct_pred2=tf.math.argmax(predict2,1)
	print(correct_pred2)

	# plot images after the attack

	fig, axes =plt.subplots(1,16,figsize=(28,28))
	axes=axes.flatten()
	for img, ax in zip(advs1, axes):
		ax.imshow(np.squeeze(img), cmap="gray")
		ax.axis("off")
	plt.tight_layout()
	plt.show()


	# # #calculate and report the robust accuracy
	# robust_accuracy = 1 - success.float32().mean(axis=-1)
	# for eps, acc in zip(epsilons, robust_accuracy):
	# 	print(eps, acc.item())



#LinfPGD()
#0.0 1.0
#0.001 1.0
#0.01 1.0
#0.03 0.9375
#0.1 0.0
#0.3 0.0
#0.5 0.0
#1.0 0.0

#FGSM()
#0.0 1.0
#0.001 1.0
#0.01 1.0
#0.03 1.0
#0.1 0.1875
#0.3 0.0
#0.5 0.0
#1.0 0.0