import numpy as np 
import csv , sys
from sklearn.decomposition import PCA       ## Source: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

with open(sys.argv[1],newline='') as csvfile:
    rows = csv.reader(csvfile)              ## Read training data
    data = []                               ## Store the data from file
    for row in rows:
        data.append(row) 
    data = data[2:]
    
    data = np.array(data)
    data = np.delete(data,2,0)              ## Missing data => remove

    data = np.delete(data,0,0)              ## Positive(attribute #4 = 2) outlier => remove 
    data = np.delete(data,4,0) 
    data = np.delete(data,5,0) 
    data = np.delete(data,6,0) 
    data = np.delete(data,11,0) 
    data = np.delete(data,2542,0) 
  
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
            mean = data[:,i].astype(np.double).mean()                   ## Normalization 
            std = data[:,i].astype(np.double).std()
            if(std == 0):                                           
                print(i)
            data[:,i] = (data[:,i].astype(np.double) - mean) / std  
        else:  
            mean = data[:,i].astype(np.double).mean()                   ## Normalization 
            std = data[:,i].astype(np.double).std()
            if(std == 0):                                           
                print(i)
            data[:,i] = (data[:,i].astype(np.double) - mean) / std  

    Y = data[:,3].reshape(-1,1).astype(np.double)                       ## Extract attribute #4 as targets
    X = np.delete(data,3,1).astype(np.double)


    print(X.shape)
    print(Y.shape)

    #sm = SMOTE(sampling_strategy = 1)                                   ## Use SMOTE to generate minor class samples    source: https://imbalanced-learn.org/en/stable/generated/imblearn.over_sampling.SMOTE.html
    #X, Y = sm.fit_resample(X, Y) 
    
    count_0 = 0
    count_1 = 0
    count_1_list = []
    for i in range(Y.shape[0]):
        if Y[i][0] == 0:
            count_0 = count_0 + 1
        else:
            count_1 = count_1 + 1
            count_1_list.append(i)
    print('count_0:',count_0)
    print('count_1:',count_1)
                                                                        ## Copy the positive (attribute #4 = 2) samples
     
    ori_one_X , ori_one_Y = X[count_1_list] , Y[count_1_list] 
    for i in range(100): 
        noise = np.random.normal(0, 0.3, ori_one_X.shape)
        add_one_X = ori_one_X + noise 
        X = np.concatenate((X,add_one_X),axis = 0)
        Y = np.concatenate((Y,ori_one_Y),axis = 0)
    '''
    number = 1500
    while(number > 0): 
        for i in range(Y.shape[0]):
            if Y[i][0] == 0:
                X = np.delete(X,i,0)
                Y = np.delete(Y,i,0)
                number = number - 1
                break 
    ''' 


    pca = PCA(n_components = 2)                                         ## Use PCA map the data onto a two-dimensional plane
    newData = pca.fit_transform(X)  

    plt.figure(figsize = (8,8))                                         ## Source: https://matplotlib.org/gallery/lines_bars_and_markers/scatter_with_legend.html
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2 component PCA') 
	
    Positive = []
    Negative = []

    for i in range(Y.shape[0]):
    	if Y[i] == 1:
    		Positive.append(newData[i])
    	else:
    		Negative.append(newData[i])
 
    Positive = np.array(Positive)
    Negative = np.array(Negative)

    print(Positive.shape)
    print(Negative.shape) 

    plt.scatter(Positive[:,0], Positive[:,1], c = 'r', s = 50 , alpha = 0.5 , label = 'Positive')  
    plt.scatter(Negative[:,0], Negative[:,1], c = 'g', s = 50 , alpha = 0.5 , label = 'Negative')  
    
    plt.legend(loc='upper right')
    plt.show()
