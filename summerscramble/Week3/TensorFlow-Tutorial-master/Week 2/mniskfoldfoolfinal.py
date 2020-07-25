import tensorflow as tf 
import eagerpy as ep 
import numpy as np 
from foolbox import TensorFlowModel, accuracy, samples
from foolbox.attacks import LinfPGD
from foolbox.attacks import BoundaryAttack
from foolbox.attacks import FGSM
from foolbox.attacks import LinfDeepFoolAttack
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import pairwise_distances
import time
#pre and advs, _, 

if __name__ == '__main__':

	mnist=tf.keras.datasets.mnist

	from sklearn.model_selection import KFold

	kf=KFold(n_splits=5)

	#kf.get_n_splits()

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

	for train_index, test_index in kf.split(x_train):
		x_trains, x_tests=x_train[train_index],x_train[test_index]
		y_trains, y_tests=y_train[train_index], y_train[test_index]
		model.fit(x_trains,y_trains, validation_data=(x_tests,y_tests), epochs=2)

	#model.fit(x_train,y_train, epochs=5)

	model.evaluate(x_test,y_test,verbose=2)

	#instantiate the model
	fmodel=TensorFlowModel(model, bounds=(0,1))

	#get data and test the model
	#wrapping the tensors with ep.astensors is optional, but it allows
	#us to work with EagerPy tensors in the following

	##########################################################
	images, labels = samples(fmodel, dataset="mnist", batchsize=100)
	images1, labels1=ep.astensors(*samples(fmodel, dataset="mnist", batchsize=100))
	print(accuracy(fmodel, images1, labels1))


	predict=fmodel(images).numpy()
	tf.nn.softmax(predict).numpy()
	correct_pred=tf.math.argmax(predict,1)
	print(correct_pred)

	#print(images)
	images_arr=np.array(images)

	#print(images_arr)
	#print(images_arr.shape) #16,28,28,1


	#plot images before attack

	# fig, axes =plt.subplots(1,16,figsize=(28,28))
	# axes=axes.flatten()
	# for img, ax in zip(images_arr, axes):
	# 	ax.imshow(np.squeeze(img), cmap="gray")
	# 	ax.axis("off")
	# plt.tight_layout()
	# fig.savefig('beforePGDattack.jpg',bbox_inches='tight', dpi=150)
	# plt.show()

	###########################################################
	#attacks1=[
		#FGSM(),
		#LinfPGD(),
		#LinfDeepFoolAttack()
	#]

	#apply the PGD attack
	attack=LinfPGD()
	epsilons=[0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
	t0=time.process_time()
	_, advsPGD, success = attack(fmodel, images, labels, epsilons=epsilons)
	t1=time.process_time()
	attacktimePGD=t1-t0
	# print("done with attack")

	#print(success)
	listsuccess=[]
	for anepval in success:
		#print(anepval)
		totalsuccess=0
		for abool in anepval:
			abool1=np.array(abool)
			#print(abool1)
			if abool1==True:
				totalsuccess+=1
		listsuccess.append(totalsuccess)

	listsuccessPGD=listsuccess
	



	#apply FGSM attack
	attack=FGSM()
	epsilons=[0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
	t0=time.process_time()
	_, advsFGSM, success = attack(fmodel, images, labels, epsilons=epsilons)
	t1=time.process_time()
	attacktimeFGSM=t1-t0
	# print("done with attack")

	#print(success)
	listsuccessFGSM=[]
	for anepval in success:
		#print(anepval)
		totalsuccess=0
		for abool in anepval:
			abool1=np.array(abool)
			#print(abool1)
			if abool1==True:
				totalsuccess+=1
		listsuccessFGSM.append(totalsuccess)

	


	#apply the DeepFool attack
	attack=LinfDeepFoolAttack()
	epsilons=[0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]

	t0=time.process_time()
	_, advsDF, success = attack(fmodel, images, labels, epsilons=epsilons)
	t1=time.process_time()

	attacktimeDF=t1-t0
	# print("done with attack")

	#print(success)
	listsuccessDF=[]
	for anepval in success:
		#print(anepval)
		totalsuccess=0
		for abool in anepval:
			abool1=np.array(abool)
			#print(abool1)
			if abool1==True:
				totalsuccess+=1
		listsuccessDF.append(totalsuccess)


	foundexPGD=sum(listsuccessPGD)
	foundexFGSM=sum(listsuccessFGSM)
	foundexDF=sum(listsuccessDF)


	print("PGD: ", listsuccessPGD)
	print("PGD sum: ", foundexPGD)

	print("FGSM: ", listsuccessFGSM)
	print("FGSM sum: ", foundexFGSM)

	print("DF: ", listsuccessDF)
	print("DF sum: ", foundexDF)


	print("Time it took for PGD attack: ", attacktimePGD)
	print("Time it took for FGSM attack: ", attacktimeFGSM)
	print("Time it took for DF attack: ", attacktimeDF)

	
	PGDavg=attacktimePGD/foundexPGD
	FGSMavg=attacktimeFGSM/foundexFGSM
	DFavg=attacktimeDF/foundexDF
	print("Average Time to find a PGD advs ex:", PGDavg)
	print("Average Time to find a FGSM advs ex:", FGSMavg)
	print("Average Time to find a Deepfool advs ex:", DFavg)


	#advs2=advs1[7] #should be size 100, 28, 28, 1
	#print(advs2)

	# tried to compare rows, did not like 1d array
	# bigmetric=[]
	# for i in range(len(images_arr)):
	# 	themetric=[]
	# 	for j in range(len(images_arr[i])):
	# 		therowmetric=pairwise_distances(np.squeeze(images_arr[i][j]), np.squeeze(advs1[i][j]), metric='l2')
	# 		themetric.append(therowmetric)
	# 	bigmetric.append(np.average(themetric))
	# print(themetric)
	# print(bigmetric)

	advs1=np.array(advsPGD) #8,16,28,28,1 now 8,100,28,28,1
	print("done with arrays")
	finallistl2PGD=[]
	for eps in range(len(epsilons)):
		advs2=advs1[eps]
		bigmetric=[]
		for i in range(len(images_arr)):
			goodmetric=[]
			themetric=pairwise_distances(np.squeeze(images_arr[i]), np.squeeze(advs2[i]), metric='l2')
			#print(advs2.size())
			for j in range(len(images_arr[i])):
					goodmetric.append(themetric[j][j])
			bigmetric.append(np.average(goodmetric))
		#print(goodmetric)
		#print(bigmetric) 
		finallistl2PGD.append(np.average(bigmetric))

	advs1=np.array(advsFGSM) #8,16,28,28,1 now 8,100,28,28,1
	print("done with arrays")
	finallistl2FGSM=[]
	for eps in range(len(epsilons)):
		advs2=advs1[eps]
		bigmetric=[]
		for i in range(len(images_arr)):
			goodmetric=[]
			themetric=pairwise_distances(np.squeeze(images_arr[i]), np.squeeze(advs2[i]), metric='l2')
			#print(advs2.size())
			for j in range(len(images_arr[i])):
					goodmetric.append(themetric[j][j])
			bigmetric.append(np.average(goodmetric))
		#print(goodmetric)
		#print(bigmetric) 
		finallistl2FGSM.append(np.average(bigmetric))


	advs1=np.array(advsDF) #8,16,28,28,1 now 8,100,28,28,1
	print("done with arrays")
	finallistl2Df=[]
	for eps in range(len(epsilons)):
		advs2=advs1[eps]
		bigmetric=[]
		for i in range(len(images_arr)):
			goodmetric=[]
			themetric=pairwise_distances(np.squeeze(images_arr[i]), np.squeeze(advs2[i]), metric='l2')
			#print(advs2.size())
			for j in range(len(images_arr[i])):
					goodmetric.append(themetric[j][j])
			bigmetric.append(np.average(goodmetric))
		#print(goodmetric)
		#print(bigmetric) 
		finallistl2Df.append(np.average(bigmetric))


	print("L2 avgs for each eps value PGD: ", finallistl2PGD)
	print("L2 avgs for each eps value FGSM: ", finallistl2FGSM)
	print("L2 avgs for each eps value DF: ", finallistl2Df)

	x=PrettyTable()

	x.field_names=['Attacks', 'Epsilon 0', 'Epsilon 0.001', 'Epsilon 0.01', 'Epsilon 0.03', 'Epsilon 0.1', 'Epsilon 0.3', 'Epsilon 0.5', 'Epsilon 1.0']

	x.add_row(['PGD: Advs Ex Found', listsuccessPGD[0], listsuccessPGD[1], listsuccessPGD[2], listsuccessPGD[3], listsuccessPGD[4], listsuccessPGD[5], listsuccessPGD[6], listsuccessPGD[7]])
	x.add_row(['FGSM: Advs Ex Found', listsuccessFGSM[0], listsuccessFGSM[1], listsuccessFGSM[2], listsuccessFGSM[3], listsuccessFGSM[4], listsuccessFGSM[5], listsuccessFGSM[6], listsuccessFGSM[7]])
	x.add_row(['DeepFool: Advs Ex Found', listsuccessDF[0], listsuccessDF[1], listsuccessDF[2], listsuccessDF[3], listsuccessDF[4], listsuccessDF[5], listsuccessDF[6], listsuccessDF[7]])

	x.add_row(['PGD: L2 average', finallistl2PGD[0], finallistl2PGD[1], finallistl2PGD[2], finallistl2PGD[3], finallistl2PGD[4], finallistl2PGD[5], finallistl2PGD[6], finallistl2PGD[7]])
	x.add_row(['FGSM: L2 average', finallistl2FGSM[0], finallistl2FGSM[1], finallistl2FGSM[2], finallistl2FGSM[3], finallistl2FGSM[4], finallistl2FGSM[5], finallistl2FGSM[6], finallistl2FGSM[7]])
	x.add_row(['DeepFool: L2 average', finallistl2Df[0], finallistl2Df[1], finallistl2Df[2], finallistl2Df[3], finallistl2Df[4], finallistl2Df[5], finallistl2Df[6], finallistl2Df[7]])
	print(x)

	y=PrettyTable()
	y.field_names=['Attack', 'Average time to find Advs Ex']
	y.add_row(['PGD:', PGDavg])
	y.add_row(['FGSM:', FGSMavg])
	y.add_row(['DeepFool:', DFavg])

	print(y)





	#plotting the results


	somlabels=['0.0', '0.001', '0.01', '0.03', '0.1', '0.3', '0.5', '1.0']
	# plot for advs ex found

	pgd_numsuccess=listsuccessPGD
	fgsm_numsuccess=listsuccessFGSM
	df_numsuccess=listsuccessDF

	x=np.arange(len(somlabels))
	width=0.25

	fig, ax=plt.subplots()
	rects1=ax.bar(x, pgd_numsuccess, width, label='PGD')
	rects2=ax.bar(x+0.25, fgsm_numsuccess, width, label='FGSM')
	rects3=ax.bar(x+0.5, df_numsuccess, width, label='DeepFool')


	ax.set_ylabel('Number of Successful Advs Ex Found')
	ax.set_title('Advs Examples Found by different Attacks')
	ax.set_xticks(x)
	ax.set_xticklabels(somlabels)
	ax.legend()

	fig.tight_layout()
	plt.show()


	#plot for L2 

	x=np.arange(len(somlabels))
	width=0.25

	fig,ax=plt.subplots()
	rects1=ax.bar(x, finallistl2PGD, width, label='PGD')
	rects2=ax.bar(x+0.25, finallistl2FGSM, width, label='FGSM')
	rects3=ax.bar(x+0.5, finallistl2Df, width, label='DeepFool')

	ax.set_ylabel('Avg L2')
	ax.set_xlabel('Epsilon Values')
	ax.set_title('Avg L2 for each Epsilon value for different Attacks')
	ax.set_xticks(x)
	ax.set_xticklabels(somlabels)
	ax.legend()


	fig.tight_layout()
	plt.show()



	#plot for time
	timelabels=['PGD', 'FGSM', 'DeepFool']
	x=np.arange(len(timelabels))
	width=0.25

	fig,ax=plt.subplots()
	times=[PGDavg,FGSMavg,DFavg]
	ax.bar(timelabels, times)

	ax.set_ylabel('Time (s)')
	ax.set_xlabel('Attack')
	ax.set_title('Avg Time for each attack to find Advs Ex')

	fig.tight_layout()
	plt.show()






 

	# predict2=fmodel(advs1).numpy()
	# tf.nn.softmax(predict2).numpy()
	# correct_pred2=tf.math.argmax(predict2,1)
	# print(correct_pred2)


	# beforeandafterc=np.column_stack((correct_pred,correct_pred2))
	# print(beforeandafterc)

	# plot images after the attack

	# fig, axes =plt.subplots(1,16,figsize=(28,28))
	# axes=axes.flatten()
	# for img, ax in zip(advs1, axes):
	# 	ax.imshow(np.squeeze(img), cmap="gray")
	# 	ax.axis("off")
	# plt.tight_layout()
	# fig.savefig('afterPGDattack.jpg',bbox_inches='tight', dpi=150)
	# plt.show()



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

#DeepFool()
#0.0 1.0
#0.001 1.0
#0.01 1.0
#0.03 0.88
#0.1 0.06
#0.3 0.0
#0.5 0.0
#1.0 0.0


#I would recommend DeepFool for finding advs examples because it is efficient and effective