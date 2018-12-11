import numpy as np 
import csv , sys
from sklearn import tree
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

weight_positive = 0.9999999
clf = tree.DecisionTreeClassifier(class_weight = {0 : 1 - weight_positive , 1 : weight_positive})
normalize = {}

with open(sys.argv[1],newline='') as csvfile:
    rows = csv.reader(csvfile)
    data = []
    for row in rows:
        data.append(row) 
    data = data[2:]
    
    data = np.array(data)
    data = np.delete(data,2,0)       		## missing data

    data = np.delete(data,176,1)            ## these columns have std = 0
    data = np.delete(data,167,1)
    data = np.delete(data,166,1)
    data = np.delete(data,165,1)         
    data = np.delete(data,5,1)
    data = np.delete(data,4,1) 

    for i in range(data.shape[1]): 
        if i == 3 : 
            for j in range(data.shape[0]):
                if data[j][i] == '1':
                    data[j][i] = 0
                elif data[j][i] == '2':
                    data[j][i] = 1
                else:
                    print('error target')
        elif data[0][i] == 'TRUE' or data[0][i] == 'FALSE':
            for j in range(data.shape[0]):
                if data[j][i] == 'TRUE':
                    data[j][i] = 1.0
                elif data[j][i] == 'FALSE':
                    data[j][i] = 0.0
                else:
                    print(j,i,data[j][i]) 
                    print('other type')

        else:  
            mean = data[:,i].astype(np.double).mean()
            std = data[:,i].astype(np.double).std()
            if(std == 0):                                           
                print(i)
            data[:,i] = (data[:,i].astype(np.double) - mean) / std 
            normalize[i] = [mean,std]

    Y = data[:,3].reshape(-1,1).astype(np.double)
    X = np.delete(data,3,1).astype(np.double)

    sm = SMOTE(sampling_strategy = 1)
    X, Y = sm.fit_resample(X, Y)

    print(X.shape)
    print(Y.shape) 

    clf.fit(X, Y)

with open(sys.argv[2],newline='') as csvfile:
    rows = csv.reader(csvfile)
    data = []
    for row in rows:
        data.append(row) 
    data = data[2:]
    
    data = np.array(data)
    data = np.delete(data,1103,0) 
    data = np.delete(data,1102,0) 
    data = np.delete(data,699,0) 
    data = np.delete(data,5,0)              ## missing data

    data = np.delete(data,176,1)            ## these columns have std = 0
    data = np.delete(data,167,1)
    data = np.delete(data,166,1)
    data = np.delete(data,165,1)         
    data = np.delete(data,5,1)
    data = np.delete(data,4,1)
 

    for i in range(data.shape[1]): 
        if i == 3 : 
            for j in range(data.shape[0]):
                if data[j][i] == '1':
                    data[j][i] = 0
                elif data[j][i] == '2':
                    data[j][i] = 1
                else:
                    print('error target')
        elif data[0][i] == 'TRUE' or data[0][i] == 'FALSE':
            for j in range(data.shape[0]):
                if data[j][i] == 'TRUE':
                    data[j][i] = 1.0
                elif data[j][i] == 'FALSE':
                    data[j][i] = 0.0
                else:
                    print(j,i,data[j][i]) 
                    print('other type')
        else:  
            data[:,i] = (data[:,i].astype(np.double) - normalize[i][0]) / normalize[i][1]

    Y = data[:,3].reshape(-1,1).astype(np.double)
    X = np.delete(data,3,1).astype(np.double)

    t1_p1 = 0
    t1_p0 = 0
    t0_p1 = 0
    t0_p0 = 0 
    for i in range(Y.shape[0]):
        if Y[i][0] == 1 and clf.predict(X[i].reshape(1,-1)) == 1:
            t1_p1 = t1_p1 + 1
        elif Y[i][0] == 1 and clf.predict(X[i].reshape(1,-1)) == 0:
            t1_p0 = t1_p0 + 1
        elif Y[i][0] == 0 and clf.predict(X[i].reshape(1,-1)) == 1:
            t0_p1 = t0_p1 + 1
        elif Y[i][0] == 0 and clf.predict(X[i].reshape(1,-1)) == 0:
            t0_p0 = t0_p0 + 1
    print('t1_p1: ',t1_p1 , 't0_p1: ',t0_p1 )
    print('t1_p0: ',t1_p0 , 't0_p0: ',t0_p0 )

   