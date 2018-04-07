import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd, eigh
import math
from skimage.io import imread, imsave
from sklearn.metrics.pairwise import cosine_similarity 
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 
training_set = []
testing_set = []
img_shape = imread('hw1_dataset/1_1.png').shape 
for i in range(1,41):
	for j in range(1,11):
		image = imread('hw1_dataset/'+str(i)+'_'+str(j)+'.png')
		X = image.flatten()
		if(j<=6) : training_set.append(X)
		else : testing_set.append(X)
training_set = np.array(training_set)
testing_set = np.array(testing_set)  

mean_train = training_set.mean(axis = 0)
arr_train = training_set - mean_train
val_train , U_train = eigh(np.cov(arr_train.T))
U_train=U_train.T[::-1]
print(U_train[0])

# 2_a
"""
fig = plt.figure()
for i in range(3):
	eigface = U_train[i].reshape(img_shape)
	ax = fig.add_subplot(3, 3, i + 1)
	ax.imshow(eigface, cmap='gray')
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))

fig.suptitle('Eigenface')
fig.savefig('eigenface.png')

plt.figure()
plt.imshow((training_set.mean(axis=0)).reshape(img_shape), cmap='gray')
plt.suptitle('Avg Eigenface')
plt.xticks(np.array([]))
plt.yticks(np.array([]))
plt.savefig('avg_eigenface.png') 
"""
# 2_b

require = [3,50,100,239]

fig = plt.figure()
for i,j in enumerate(require): 
	re = arr_train[0].dot(U_train[:j].T).dot(U_train[:j])
	eigface = (re + mean_train).reshape(img_shape)
	mse = np.mean((re-arr_train[0])**2)
	print("{} dim error = {}".format(j,mse))
	ax = fig.add_subplot(1, 4, i + 1)
	ax.imshow(eigface, cmap='gray')
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))

fig.suptitle('Reconstruct face') 
fig.savefig('reconstruct.png')

# 2_c choose (d=3,k=1)

dim = [3,50,159]
k_near = [1,3,5]
training_set_a = []
training_set_b = []
training_set_c = []
for i in range(240):
	if(i%6==0 or i%6==1): training_set_a.append(training_set[i])
	elif(i%6==2 or i%6==3): training_set_b.append(training_set[i])
	elif(i%6==4 or i%6==5): training_set_c.append(training_set[i])
training_set_a = np.array(training_set_a)
training_set_b = np.array(training_set_b)
training_set_c = np.array(training_set_c)

answer = []
for i in range(80): 
	answer.append(math.floor(i/2))

test_answer = []
for i in range(160): 
	test_answer.append(math.floor(i/4)) 
 
mean_train_a =  (np.concatenate((training_set_b,training_set_c),axis=0)).mean(axis = 0)
arr_train_a =  (np.concatenate((training_set_b,training_set_c),axis=0)) - mean_train_a
val_train_a , U_train_a = eigh(np.cov(arr_train_a.T))
U_train_a=U_train_a.T[::-1]

mean_train_b =  (np.concatenate((training_set_a,training_set_c),axis=0)).mean(axis = 0)
arr_train_b =  (np.concatenate((training_set_a,training_set_c),axis=0)) - mean_train_b
val_train_b , U_train_b = eigh(np.cov(arr_train_b.T))
U_train_b=U_train_b.T[::-1]

mean_train_c =  (np.concatenate((training_set_a,training_set_b),axis=0)).mean(axis = 0)
arr_train_c =  (np.concatenate((training_set_a,training_set_b),axis=0)) - mean_train_c
val_train_c , U_train_c = eigh(np.cov(arr_train_c.T))
U_train_c=U_train_c.T[::-1]

for d in dim:
	for k in k_near:
		knn = KNeighborsClassifier(n_neighbors=k)
		train_data = (np.concatenate((training_set_a,training_set_b),axis=0)).dot(U_train_c[:d].T)  
		#print('train_data:',train_data.shape)
		tmp = np.array(answer) 
		#print('tmp:',tmp.shape)
		train_label = np.concatenate((tmp,tmp),axis=0) 
		#print('train_label:',train_label.shape)
		test_data = training_set_c.dot(U_train_c[:d].T)
		test_label = np.array(answer) 
		knn.fit(train_data,train_label)
		print("validation({},{},c): {}".format(d,k,knn.score(test_data,test_label)))
for d in dim:
	for k in k_near:
		knn = KNeighborsClassifier(n_neighbors=k)
		train_data = (np.concatenate((training_set_a,training_set_c),axis=0)).dot(U_train_b[:d].T) 
		tmp = np.array(answer) 
		train_label = np.concatenate((tmp,tmp),axis=0) 
		test_data = training_set_b.dot(U_train_b[:d].T)
		test_label = np.array(answer) 
		knn.fit(train_data,train_label)
		print("validation({},{},b): {}".format(d,k,knn.score(test_data,test_label)))
for d in dim:
	for k in k_near:
		knn = KNeighborsClassifier(n_neighbors=k)
		train_data = (np.concatenate((training_set_b,training_set_c),axis=0)).dot(U_train_a[:d].T) 
		tmp = np.array(answer) 
		train_label = np.concatenate((tmp,tmp),axis=0) 
		test_data = training_set_a.dot(U_train_a[:d].T)
		test_label = np.array(answer) 
		knn.fit(train_data,train_label)
		print("validation({},{},a): {}".format(d,k,knn.score(test_data,test_label)))

k = 1
d = 159
knn = KNeighborsClassifier(n_neighbors=k)
train_data = (np.concatenate((training_set_a,training_set_b,training_set_c),axis=0)).dot(U_train[:d].T)  
tmp = np.array(answer) 
train_label = np.concatenate((tmp,tmp,tmp),axis=0) 
test_data = testing_set.dot(U_train[:d].T)
test_label = np.array(test_answer) 
knn.fit(train_data,train_label)
print("score: {}".format(knn.score(test_data,test_label)))


