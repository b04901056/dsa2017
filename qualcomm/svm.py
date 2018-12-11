import numpy as np 
import csv , sys
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

weight_positive = 0.9935                    ## Make SVM cost-sensitive
normalize = {}                              ## Record the mean and standard deviation for testing set normalization
clf = SVC(gamma='auto',class_weight = {0 : 1 - weight_positive , 1 : weight_positive})  ## Source: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

with open(sys.argv[1],newline='') as csvfile:
    rows = csv.reader(csvfile)              ## Read training data
    data = []                               ## Store the data from file
    for row in rows:
        data.append(row) 
    data = data[2:]
    
    data = np.array(data)
    data = np.delete(data,2,0)       		## Missing data => remove

    data = np.delete(data,176,1)            ## These columns have std = 0 => remove
    data = np.delete(data,167,1)
    data = np.delete(data,166,1)
    data = np.delete(data,165,1)         
    data = np.delete(data,5,1)
    data = np.delete(data,4,1) 
    data = np.delete(data,166,1)  
    data = np.delete(data,165,1)  
    data = np.delete(data,164,1)  

    for i in range(data.shape[1]): 
        if i == 3 :                                                     ## Transform label of attribute #4 '2' to 1(positive), '1' to 0(negative) 
            for j in range(data.shape[0]):
                if data[j][i] == '1':
                    data[j][i] = 0
                elif data[j][i] == '2':
                    data[j][i] = 1
                else:
                    print('error target')
        elif data[0][i] == 'TRUE' or data[0][i] == 'FALSE':             ## Transform label 'TRUE' to 1, 'Negative' to 0
            for j in range(data.shape[0]):
                if data[j][i] == 'TRUE':
                    data[j][i] = 1
                elif data[j][i] == 'FALSE':
                    data[j][i] = 0
                else:
                    print(j,i,data[j][i]) 
                    print('other type')            
            mean = data[:,i].astype(np.double).mean()                   ## Normalization. Record mean and standard deviation 
            std = data[:,i].astype(np.double).std()
            if(std == 0):                                           
                print(i)
            data[:,i] = (data[:,i].astype(np.double) - mean) / std 
            normalize[i] = [mean,std]
        else:  
            mean = data[:,i].astype(np.double).mean()                   ## Normalization. Record mean and standard deviation 
            std = data[:,i].astype(np.double).std()
            if(std == 0):                                           
                print(i)
            data[:,i] = (data[:,i].astype(np.double) - mean) / std 
            normalize[i] = [mean,std]

    Y = data[:,3].reshape(-1,1).astype(np.double)                       ## Extract attribute #4 as targets
    X = np.delete(data,3,1).astype(np.double)
    
    sm = SMOTE(sampling_strategy = 1)                                   ## Use SMOTE to generate minor class samples    source: https://imbalanced-learn.org/en/stable/generated/imblearn.over_sampling.SMOTE.html
    X, Y = sm.fit_resample(X, Y) 
    #print(X.shape)
    #print(Y) 
    clf.fit(X, Y) 

with open(sys.argv[2],newline='') as csvfile:
    rows = csv.reader(csvfile)                                          ## Read testing data
    data = []                                                           ## Store the data from file
    for row in rows:
        data.append(row) 
    data = data[2:]
    
    data = np.array(data)
    data = np.delete(data,1103,0)                                       ## Missing data => remove
    data = np.delete(data,1102,0) 
    data = np.delete(data,699,0) 
    data = np.delete(data,5,0)               

    data = np.delete(data,176,1)                                        ## These columns have std = 0 => remove            
    data = np.delete(data,167,1)
    data = np.delete(data,166,1)
    data = np.delete(data,165,1)         
    data = np.delete(data,5,1)
    data = np.delete(data,4,1)
    data = np.delete(data,166,1)  
    data = np.delete(data,165,1)  
    data = np.delete(data,164,1) 
 

    for i in range(data.shape[1]):                                                          ## Transform label of attribute #4 '2' to 1(positive), '1' to 0(negative) 
        if i == 3 : 
            for j in range(data.shape[0]):
                if data[j][i] == '1':
                    data[j][i] = 0
                elif data[j][i] == '2':
                    data[j][i] = 1
                else:
                    print('error target')
        elif data[0][i] == 'TRUE' or data[0][i] == 'FALSE':                                 ## Transform label 'TRUE' to 1, 'Negative' to 0
            for j in range(data.shape[0]):
                if data[j][i] == 'TRUE':
                    data[j][i] = 1.0
                elif data[j][i] == 'FALSE':
                    data[j][i] = 0.0
                else:
                    print(j,i,data[j][i]) 
                    print('other type')
            data[:,i] = (data[:,i].astype(np.double) - normalize[i][0]) / normalize[i][1]   ## Normalization 
        else:  
            data[:,i] = (data[:,i].astype(np.double) - normalize[i][0]) / normalize[i][1]   ## Normalization 

    Y = data[:,3].reshape(-1,1).astype(np.double)                                           ## Extract attribute #4 as targets
    X = np.delete(data,3,1).astype(np.double)

    t1_p1 = 0                                                                               ## Confusion matrix initialization
    t1_p0 = 0
    t0_p1 = 0
    t0_p0 = 0  
   
    for i in range(Y.shape[0]):                                                             ## Calculate confusion matrix
        if Y[i][0] == 1 and clf.predict(X[i].reshape(1,-1)) == 1:
            t1_p1 = t1_p1 + 1
        elif Y[i][0] == 1 and clf.predict(X[i].reshape(1,-1)) == 0:
            t1_p0 = t1_p0 + 1
        elif Y[i][0] == 0 and clf.predict(X[i].reshape(1,-1)) == 1:
            t0_p1 = t0_p1 + 1
        elif Y[i][0] == 0 and clf.predict(X[i].reshape(1,-1)) == 0:
            t0_p0 = t0_p0 + 1

    print('t1_p1: ',t1_p1 , 't0_p1: ',t0_p1 )                                               ## Print confusion matrix
    print('t1_p0: ',t1_p0 , 't0_p0: ',t0_p0 )

   