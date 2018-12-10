import numpy as np 
import csv , sys
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt

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

    Y = data[:,3].reshape(-1,1).astype(np.double)
    X = np.delete(data,3,1).astype(np.double)

    print(X.shape)
    print(Y.shape)

    pca=PCA(n_components = 128)
    newData=pca.fit_transform(X)

    print(newData.shape)
    print(newData)

    plt.figure(figsize = (8,8)) 
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2 component PCA') 
	
    Positive = []
    Negative = []

    for i in range(Y.shape[0]):
    	if Y[i][0] == 1:
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
