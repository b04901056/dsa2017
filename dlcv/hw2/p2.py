import numpy as np
import matplotlib as plt
from matplotlib.pyplot import imshow
from sklearn import cluster, datasets
from PIL import Image
from skimage import io, color
import scipy.io as sio
from scipy import signal

## 2-a-1
'''
zebra = Image.open('Problem2/zebra.jpg')
mountain = Image.open('Problem2/mountain.jpg')
print(zebra)
zebra = np.array(zebra).reshape(-1,3)
mountain = np.array(mountain).reshape(-1,3)


kmeans_fit = cluster.KMeans(n_clusters = 10,max_iter=1000)
cluster_labels = kmeans_fit.fit_predict(zebra)
result = np.array(cluster_labels).reshape(331,640)
im = plt.pyplot.imshow(result)
plt.pyplot.colorbar(im, orientation='horizontal')
plt.pyplot.show()
'''

## 2-a-2
'''
rgb = io.imread('Problem2/mountain.jpg')
#zebra = color.rgb2lab(rgb).reshape(-1,3)
mountain = color.rgb2lab(rgb).reshape(-1,3)
#print(lab)
#print(lab.shape)
#input()

kmeans_fit = cluster.KMeans(n_clusters = 10,max_iter=1000)
cluster_labels = kmeans_fit.fit_predict(mountain)
result = np.array(cluster_labels).reshape(417,640)
im = plt.pyplot.imshow(result)
plt.pyplot.colorbar(im, orientation='horizontal')
plt.pyplot.show()
'''

## 2-b-1
'''
zebra = io.imread('Problem2/zebra.jpg', as_grey=True).reshape(331,640)
mountain = io.imread('Problem2/mountain.jpg', as_grey=True).reshape(417,640)
print('zebra.shape ',zebra.shape)
bank = sio.loadmat('Problem2/filterBank.mat')
bank = np.array(bank['F']).reshape(38,49,49)
print('bank[0].shape ',bank[0].shape)
 
response = []
for i in range(38): 
    print(i)
    corr = signal.correlate2d(mountain, bank[i], boundary='symm', mode='same')
    response.append(corr)

result = np.dstack(response)
result = result.reshape(-1,38)
 
kmeans_fit = cluster.KMeans(n_clusters=6,max_iter=1000)
cluster_labels = kmeans_fit.fit_predict(result)
result = np.array(cluster_labels).reshape(417,640)
im = plt.pyplot.imshow(result)
plt.pyplot.colorbar(im, orientation='horizontal')
plt.pyplot.show()
'''

## 2-b-2


rgb = io.imread('Problem2/mountain.jpg')
#zebra_3 = color.rgb2lab(rgb).reshape(331,640,3)
mountain_3 = color.rgb2lab(rgb).reshape(417,640,3)

#zebra = Image.open('Problem2/zebra.jpg')
#mountain = Image.open('Problem2/mountain.jpg')
#zebra_3 = np.array(zebra).reshape(331,640,3)
#mountain_3 = np.array(mountain).reshape(417,640,3)

zebra = io.imread('Problem2/zebra.jpg', as_grey=True).reshape(331,640)
mountain = io.imread('Problem2/mountain.jpg', as_grey=True).reshape(417,640)
bank = sio.loadmat('Problem2/filterBank.mat')
bank = np.array(bank['F']).reshape(38,49,49)
 
response = []
'''
for i in range(38): 
    print(i)
    corr = signal.correlate2d(zebra, bank[i], boundary='symm', mode='same')
    response.append(corr) 
result = np.dstack(response)
zebra_38 = result.reshape(331,640,38)
'''
for i in range(38):             
    print(i)                    
    corr = signal.correlate2d(mountain, bank[i], boundary='symm', mode='same')
    response.append(corr)       
result = np.dstack(response)    
mountain_38 = result.reshape(417,640,38)

#zebra = np.concatenate((zebra_3,zebra_38),axis=2)
mountain = np.concatenate((mountain_3,mountain_38),axis=2)

#zebra = zebra.reshape(-1,41)
mountain = mountain.reshape(-1,41)

kmeans_fit = cluster.KMeans(n_clusters=6,max_iter=1000)
cluster_labels = kmeans_fit.fit_predict(mountain)
result = np.array(cluster_labels).reshape(417,640)  
im = plt.pyplot.imshow(result)
plt.pyplot.colorbar(im, orientation='horizontal')
plt.pyplot.show()





















