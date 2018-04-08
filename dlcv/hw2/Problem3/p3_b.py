import cv2
import os
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize
from scipy.spatial import distance
from sklearn import neighbors
import argparse

parser = argparse.ArgumentParser(description='setting parameter.')
parser.add_argument('-trp','--train_path', dest='train_path',type=str,required=True)
parser.add_argument('-tep','--test_path', dest='test_path',type=str,required=True)
parser.add_argument('-func', dest='func',type=str,required=True)
parser.add_argument('-int','--num_interest', dest='num_interest',type=int,required=True)
parser.add_argument('-clu','--cluster', dest='cluster',type=int,required=True)
parser.add_argument('-iter','--iteration', dest='iteration',type=int,required=True)
parser.add_argument('-cat', dest='cat',type=str,required=False)

args = parser.parse_args()

feature = []
train = []
train_x = []
train_y = []

##read image

for doc in os.listdir(args.train_path):
    full_path = os.path.join(args.train_path, doc)
    for image in os.listdir(full_path):
        image = os.path.join(full_path,image)
        #print('read in image_name:',image)
        img = cv2.imread(image)
        #gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(img,None) 
        kp = kp[:args.num_interest]
        des = des[:args.num_interest]
        feature.append(des)
        train.append(des)
        train_y.append(str(doc))
 
feature = np.concatenate(feature,axis=0)
p = np.random.permutation(len(feature))       
feature = feature[p]
#print('feature.shape: ',feature.shape)

km_cluster = KMeans(n_clusters=args.cluster, max_iter=args.iteration).fit(feature)
result = km_cluster.predict(feature)
#print('len(result): ',len(result))  
               
k_class = [ [] for i in range(args.cluster)]

for i in range(len(result)):
    k_class[result[i]].append(feature[i].reshape(1,128))

for i in range(args.cluster):
    k_class[i] = np.concatenate(k_class[i],axis=0)

#visual_word = []
#for i in range(args.cluster):
    #visual_word.append(np.mean(k_class[i],axis=0).reshape(1,-1))
#visual_word = km_cluster.cluster_centers_
#print(visual_word.shape)
#input()
####p2-b#################################################
'''
pca = PCA(n_components=3)  
pca.fit(feature)
sub_visual_word = []

for i in range(50):
    sub_visual_word.append(pca.transform(visual_word[i])) 

for i in range(50):
    k_class[i] = pca.transform(k_class[i])

X = np.concatenate((k_class[0],k_class[1],k_class[2],k_class[3],k_class[4],k_class[5]),axis=0)
for i in range(6):
    X =np.concatenate((X,sub_visual_word[i]),axis=0)
print('X.shape: ',X.shape)
labels = []
for i in range(6):
    for j in range(k_class[i].shape[0]):
        if(i==0): labels.append('r')
        elif(i==1): labels.append('g')
        elif(i==2): labels.append('b')
        elif(i==3): labels.append('y')
        elif(i==4): labels.append('m')
        elif(i==5): labels.append('c')
for i in range(6):
    labels.append('k')
labels = np.array(labels)
print('labels: ',labels)
print('labels.shape: ',labels.shape)


fignum = 1
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(X[:, 0], X[:, 1], X[:, 2],c=labels, edgecolor='k',alpha=1)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.dist = 12
plt.show()
'''
##########################################################
def cal_dist(feature):
    return km_cluster.transform(feature) 

def cal_hard_sum(feature,ip,img):
    result = []
    dist = cal_dist(feature)
    for i in range(img):
        tmp = np.zeros((ip,args.cluster))
        count = 0
        for j in range(i*ip,(i+1)*ip):
            index = np.argmin(dist[j])
            tmp[count][index] = 1
            count += 1
        tmp = np.sum(tmp,axis=0).reshape(1,-1)
        result.append(tmp)
    result_hard_sum = np.concatenate(result,axis=0)
    result_hard_sum = normalize(result_hard_sum,norm='l2')
    return result_hard_sum

def cal_soft_sum(feature,ip,img):
    result = []
    d = cal_dist(feature)
    d = np.reciprocal(d)
    d = normalize(d,norm='l2')

    for i in range(img):
        result.append(np.sum(d[i*ip:(i+1)*ip],axis=0).reshape(1,-1))
    result = np.concatenate(result,axis=0)
    result_soft_sum = normalize(result,norm='l2') 
    return result_soft_sum
    

## soft-max
def cal_soft_max(feature,ip,img):
    result = []
    d = cal_dist(feature)
    d = np.reciprocal(d)
    d = normalize(d,norm='l2')

    for i in range(img):
        tmp = np.amax(d[i*ip:(i+1)*ip],axis=0).reshape(1,-1)
        result.append(tmp)
    result_soft_max = np.concatenate(result,axis=0)
    return result_soft_max

for x in train:
    if(args.func=='hard_sum'): des = cal_hard_sum(x,x.shape[0],1)
    elif(args.func=='soft_sum'): des = cal_soft_sum(x,x.shape[0],1)
    elif(args.func=='soft_max'): des = cal_soft_max(x,x.shape[0],1)
    train_x.append(des)

p = np.random.permutation(len(train_x))       
train_x = np.concatenate(train_x,axis=0).reshape(len(train_x),args.cluster)[p]         
train_y = np.array(train_y).reshape(-1,1).ravel()[p]

## plot histogram
print(train_x.shape)
if(args.cat=='coast'): X = train_x[1].tolist()
elif(args.cat=='suburb'): X = train_x[11].tolist()
elif(args.cat=='forest'): X = train_x[21].tolist()
elif(args.cat=='mountain'): X = train_x[31].tolist()
elif(args.cat=='highway'): X = train_x[41].tolist()
fig = plt.figure()
ax = fig.add_subplot(111)

N = args.cluster

ind = np.arange(N)               
width = 0.35                     

rects1 = ax.bar(ind, X , width,         ###change###
                color='blue',
                error_kw=dict(elinewidth=2,ecolor='red'))

ax.set_xlim(-width,len(ind)+width)
ax.set_ylim(0,1)
ax.set_xlabel('Dimension')
#ax.set_ylabel('Magnitude')
ax.set_title(args.func+'_'+args.cat)         ###change###
xTickMarks = [str(i) for i in range(args.cluster)]
ax.set_xticks(ind+width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=45, fontsize=10)
plt.savefig('p3_result/'+args.func+'_'+args.cat+'.png')
#plt.show()
'''

##training and testing

test_x = []
test_y = []

for doc in os.listdir(args.test_path):
    full_path = os.path.join(args.test_path, doc)
    for image in os.listdir(full_path):
        image = os.path.join(full_path,image)
        #print('read in image_name:',image)
        img = cv2.imread(image)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(gray,None) 
        kp = kp[:args.num_interest]
        des = des[:args.num_interest]
        if(args.func=='hard_sum'): des = cal_hard_sum(des,des.shape[0],1)
        elif(args.func=='soft_sum'): des = cal_soft_sum(des,des.shape[0],1)                                                                                                                                  
        elif(args.func=='soft_max'): des = cal_soft_max(des,des.shape[0],1)
        test_x.append(des)
        test_y.append(str(doc)) 



test_x = np.concatenate(test_x,axis=0).reshape(500,args.cluster)
test_y = np.array(test_y).reshape(-1,1).ravel()



# 建立分類器
clf = neighbors.KNeighborsClassifier()
image_clf = clf.fit(train_x, train_y)

# 預測
test_y_predict = image_clf.predict(test_x)

count = 0
for i in range(500):
    if(test_y_predict[i]==test_y[i]):
        count+=1
    #else:
        #print(test_y_predict[i],'  ',test_y[i])
print(args.train_path+' '+args.test_path+' '+args.func+' '+str(args.num_interest)+' '+str(args.cluster)+' '+str(args.iteration)+' '+'accu: ',float(count/500))

'''






























































































































































































