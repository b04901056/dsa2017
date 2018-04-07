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

train_path = 'train-100'
test_path = 'test-100'

feature = []
for doc in os.listdir(train_path):
    full_path = os.path.join(train_path, doc)
    for image in os.listdir(full_path):
        image = os.path.join(full_path,image)
        print('read in image_name:',image)
        img = cv2.imread(image)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(gray,None) 
        if(des.shape[0]<30): print('!!!!!')
        kp = kp[:30]
        des = des[:30]
        feature.append(des)

feature = np.concatenate(feature,axis=0)
#print('feature: ',feature)
p = np.random.permutation(len(feature))       
feature = feature[p]

print('feature.shape: ',feature.shape)

np.save('feature.npy',feature)


feature = np.load('feature.npy')
feature = normalize(feature, axis=0)
km_cluster = KMeans(n_clusters=50, max_iter=5000,n_init=40)
result = km_cluster.fit_predict(feature)

print('feature.shape: ',feature.shape)
result = np.array(result).reshape(-1,1)
          
print('result.shape: ',result.shape)  
merge = np.concatenate((feature,result),axis=1)
               
k_class = [ [] for i in range(50)]

for i in range(merge.shape[0]):
    k_class[ int(merge[i][128])].append(merge[i][:128].reshape(1,128))

for i in range(50):
    k_class[i] = np.concatenate(k_class[i],axis=0)

visual_word = []
for i in range(50):
    visual_word.append(np.mean(k_class[i],axis=0).reshape(1,-1))

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
    table = []
    for i in range(50):
        dist = distance.cdist(feature,visual_word[i],'euclidean')
        table.append(dist)
    dist = np.concatenate(table,axis=1)
    return dist

def cal_hard_sum(feature,ip,img):
    result = []
    dist = cal_dist(feature)
    for i in range(img):
        tmp = np.zeros((ip,50))
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

train_x = []
train_y = []
  
for doc in os.listdir(train_path):
    full_path = os.path.join(train_path, doc)
    for image in os.listdir(full_path):
        image = os.path.join(full_path,image)
        print('read in image_name:',image)
        img = cv2.imread(image)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(gray,None)
        kp = kp[:30]
        des = des[:30]
        des = cal_hard_sum(des,des.shape[0],1) ###change###
        train_x.append(des)
        train_y.append(str(doc))

p = np.random.permutation(len(train_x))       
train_x = np.concatenate(train_x,axis=0)[p]         
train_y = np.array(train_y).reshape(-1,1).ravel()[p]

print('train_x.shape" ',train_x.shape)
'''
print('feature.shape: ',feature.shape)

result_hard_sum = cal_hard_sum(feature,30,)
print('result_hard_sum.shape: ',result_hard_sum.shape)

result_soft_sum = cal_soft_sum(feature,30,feature.shape[0])
iprint('result_soft_sum.shape: ',result_soft_sum.shape) 

result_soft_max = cal_soft_max(feature,30,feature.shape[0])
print('result_soft_max.shape: ',result_soft_max.shape) 
'''



## plot histogram

hard_sum = train_x[0].tolist() ###change###
soft_sum = train_x[0].tolist()
soft_max = train_x[0].tolist()

fig = plt.figure()
ax = fig.add_subplot(111)

N = 50

ind = np.arange(N)               
width = 0.35                     

## the bars
rects1 = ax.bar(ind, soft_max, width,         ###change###
                color='black',
                error_kw=dict(elinewidth=2,ecolor='red'))

ax.set_xlim(-width,len(ind)+width)
ax.set_ylim(0,1)
ax.set_xlabel('Dimension')
ax.set_ylabel('Magnitude')
ax.set_title('Soft_max')         ###change###
xTickMarks = [str(i) for i in range(50)]
ax.set_xticks(ind+width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=45, fontsize=10)
plt.show()


##training and testing

test_x = []
test_y = []

for doc in os.listdir(test_path):
    full_path = os.path.join(test_path, doc)
    for image in os.listdir(full_path):
        image = os.path.join(full_path,image)
        print('read in image_name:',image)
        img = cv2.imread(image)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
        sift = cv2.xfeatures2d.SIFT_create()
        kp,des = sift.detectAndCompute(gray,None) 
        kp = kp[:30]
        des = des[:30]
        des = cal_soft_max(des,des.shape[0],1) ###change###
        test_x.append(des)
        test_y.append(str(doc)) 

test_x = np.concatenate(test_x,axis=0).reshape(500,50)
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
    else:
        print(test_y_predict[i],'  ',test_y[i])
print('accu: ',float(count/500))
































































































































































































