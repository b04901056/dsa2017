import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F 
import time
import numpy as np
assert F
import csv,random
from imblearn.over_sampling import SMOTE 

class Datamanager():
    def __init__(self):
        self.dataset = {} 
        self.threshold = 1e-10              ## The threshold for classifying the label of the sample after simoid function 
        self.normalize = {}                 ## Record the mean and standard deviation for testing set normalization 

        self.over_sample = False            ## Determine whether copy the positive (attribute #4 = 2) samples
        self.over_sample_rate = 150         ## Number of copies

        self.down_sample = False            ## Determine whether delete the negative (attribute #4 = 1) samples
        self.down_sample_rate = 2           ## Make number of negative samples = down_sample_rate * Number of positive samples
 
        self.smote = True                   ## Determine whether use SMOTE to generate minor class samples

        self.weighted_loss = True           ## Determine whether adjust the weight of BCE loss function
        self.weighted_loss_rate = 0.1       ## Weight of negative samples in loss function ( 1 - weighted_loss_rate for positive samples)

    def get_data(self,name,file_name,b_size,args,shuf=True):     
        with open(file_name,newline='') as csvfile:
            rows = csv.reader(csvfile)                                                          ## Read file
            data = []                                                                           ## Store the data from file
            for row in rows:
                data.append(row) 
            data = data[2:]
            
            data = np.array(data)
            if name == 'train' :                                                                ## Missing data => remove
                data = np.delete(data,2,0)            
            if name == 'test' :  
                data = np.delete(data,1103,0) 
                data = np.delete(data,1102,0) 
                data = np.delete(data,699,0) 
                data = np.delete(data,5,0) 

            data = np.delete(data,176,1)                                                        ## These columns have std = 0 => remove
            data = np.delete(data,167,1)
            data = np.delete(data,166,1)
            data = np.delete(data,165,1)         
            data = np.delete(data,5,1)
            data = np.delete(data,4,1)
            data = np.delete(data,166,1)  
            data = np.delete(data,165,1)  
            data = np.delete(data,164,1) 
 
            for i in range(data.shape[1]):           
                if i == 3 : 
                    for j in range(data.shape[0]):                                              ## Transform label '2' to 1(positive), '1' to 0(negative) 
                        if data[j][i] == '1':
                            data[j][i] = 0
                        elif data[j][i] == '2':
                            data[j][i] = 1
                        else:
                            print('error target')
                elif data[0][i] == 'TRUE' or data[0][i] == 'FALSE': #                           ## Transform label 'TRUE' to 1, 'Negative' to 0
                    for j in range(data.shape[0]):
                        if data[j][i] == 'TRUE':
                            data[j][i] = 1.0
                        elif data[j][i] == 'FALSE':
                            data[j][i] = 0.0
                        else:
                            print(j,i,data[j][i]) 
                            print('other type')
                    mean = data[:,i].astype(np.double).mean()                                   ## Normalization. Record mean and standard deviation 
                    std = data[:,i].astype(np.double).std()
                    if(std == 0):                                           
                        print(i)
                    data[:,i] = (data[:,i].astype(np.double) - mean) / std 
                    self.normalize[i] = [mean,std]

                else: 
                    if name == 'train':                                                         ## Normalization. Record mean and standard deviation 
                        mean = data[:,i].astype(np.double).mean()
                        std = data[:,i].astype(np.double).std()
                        if(std == 0):                                           
                            print(i)
                        data[:,i] = (data[:,i].astype(np.double) - mean) / std 
                        self.normalize[i] = [mean,std]
                    else:
                        data[:,i] = (data[:,i].astype(np.double) - self.normalize[i][0]) / self.normalize[i][1] 
                       


            if name == 'train' :                                                             
                np.random.shuffle(data)                                                         
                Y = data[:int(data.shape[0]*0.9),3].reshape(-1,1).astype(np.double)             ## Split training and validation set, and extract attribute #4 as targets
                Y_val = data[int(data.shape[0]*0.9):,3].reshape(-1,1).astype(np.double)
                X = np.delete(data,3,1).astype(np.double)[:int(data.shape[0]*0.9),:] 
                X_val = np.delete(data,3,1).astype(np.double)[int(data.shape[0]*0.9):,:] 

                if self.over_sample or self.down_sample or self.smote:
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
                    if self.over_sample:                                                        ## Copy the positive (attribute #4 = 2) samples
                        add_one_X , add_one_Y = X[count_1_list] , Y[count_1_list] 
                        noise = np.random.normal(0, 0.3, add_one_X.shape)
                        add_one_X = add_one_X + noise 
                        for i in range(self.over_sample_rate):
                            X = np.concatenate((X,add_one_X),axis = 0)
                            Y = np.concatenate((Y,add_one_Y),axis = 0)

                    if self.down_sample:                                                        ## Delete the negative (attribute #4 = 1) samples
                        number = int(count_0 - count_1 * (self.over_sample_rate + 1) * self.down_sample_rate)
                        while(number > 0): 
                            for i in range(Y.shape[0]):
                                if Y[i][0] == 0:
                                    X = np.delete(X,i,0)
                                    Y = np.delete(Y,i,0)
                                    number = number - 1
                                    break 
                    if self.smote:                                                              ## Use SMOTE to generate minor class(positive) samples
                        sm = SMOTE(sampling_strategy = 1)
                        X, Y = sm.fit_resample(X, Y)
                        Y = Y.reshape(-1,1) 
                        count_0 = 0
                        count_1 = 0
                        for i in range(Y.shape[0]):
                            if Y[i][0] == 0:
                                count_0 = count_0 + 1
                            else:
                                count_1 = count_1 + 1 
                        print('count_0:',count_0)
                        print('count_1:',count_1)

                print(X.shape)
                print(Y.shape) 

                X,Y = torch.from_numpy(X).cuda(),torch.from_numpy(Y).cuda()                                         ## Convert numpy array to tensor for Pytorch
                train_dataset = Data.TensorDataset(data_tensor=X[:], target_tensor=Y[:])                            ## Wrap up the input/target tensor into TensorDataset   source: https://pytorch.org/docs/stable/data.html
                self.dataset['train'] = Data.DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=shuf)     ## Put the TensorDataset in Dataloader (stored in a dictionary), shuffling the samples    source: https://pytorch.org/docs/stable/data.html

                X_val,Y_val = torch.from_numpy(X_val).cuda(),torch.from_numpy(Y_val).cuda()
                val_dataset = Data.TensorDataset(data_tensor=X_val[:], target_tensor=Y_val[:]) 
                self.dataset['val'] = Data.DataLoader(dataset=val_dataset, batch_size=b_size, shuffle=shuf) 
 
            elif name == 'test':                                                                                    ## Process the testing set
                Y = data[:,3].reshape(-1,1).astype(np.double) 
                X = np.delete(data,3,1).astype(np.double)

                X,Y = torch.from_numpy(X).cuda(),torch.from_numpy(Y).cuda()
                train_dataset = Data.TensorDataset(data_tensor=X[:], target_tensor=Y[:]) 
                self.dataset['test'] = Data.DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=shuf)      ## Put the TensorDataset in Dataloader (stored in a dictionary), not shuffling the samples    source: https://pytorch.org/docs/stable/data.html
    
    def train(self,model,trainloader,epoch):                                            ## Train the model
        model.train()                                                                   ## Set to training mode
        optimizer = torch.optim.Adam(model.parameters())                                ## Use Adam optimizer to optimize all DNN parameters    source: https://pytorch.org/docs/stable/optim.html  
        loss_func = nn.BCELoss()                                                        ## Use binary cross entropoy for model evaluation       source: https://pytorch.org/docs/stable/nn.html
        total_loss = 0                                                                  ## Calculate total loss in a epoch
        t1_p1 = 0                                                                       ## Confusion matrix initialization
        t1_p0 = 0
        t0_p1 = 0
        t0_p0 = 0

        for batch_index, (x, y) in enumerate(trainloader):                              ## Process a batch of data in each timestep
            x, y= Variable(x).cuda(), Variable(y).cuda() 
            output = model(x)                                                           ## Use present model to forecast the the result 
            if self.weighted_loss:                                                      ## Adjust the weight of BCE loss functional             source: https://pytorch.org/docs/stable/nn.html
                weight = np.empty([len(x)])
                for i in range(len(x)):
                    weight[i] = self.weighted_loss_rate
                arr = np.where(y.data == 1)
                weight[arr[0].tolist()] = 1 - self.weighted_loss_rate 
                weight = torch.from_numpy(weight).cuda().double().view(len(x),1) 
                loss_func = nn.BCELoss(weight = weight)

            loss = loss_func(output,y) 
            optimizer.zero_grad()                                                       ## Set the gradient in the previous time step to zero
            loss.backward()                                                             ## Back propagate    source: https://pytorch.org/docs/stable/optim.html
            optimizer.step()                                                            ## Gradient descent    source: https://pytorch.org/docs/stable/autograd.html
            if batch_index % 4 == 0:                                                    ## Print model status    source: https://pytorch.org/docs/stable/optim.html
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)]\t '.format(
                        epoch, batch_index * len(x), len(trainloader.dataset),
                        100. * batch_index / len(trainloader)),end='')

            total_loss+= loss.data[0]*len(x)                                            ## Sum up batch loss 

            pred = np.empty([len(x),1])                                                 ## Calculate confusion matrix
            output = output.cpu().data.numpy()
           
            for i in range(len(x)):  
                if(output[i] > self.threshold):
                    pred[i,0] = 1
                else:
                    pred[i,0] = 0

            y = y.cpu().data.numpy() 
             
            for i in range(pred.shape[0]): 
                if y[i] == 1 and pred[i] == 1:
                    t1_p1 = t1_p1 + 1
                elif y[i] == 1 and pred[i] == 0:
                    t1_p0 = t1_p0 + 1
                elif y[i] == 0 and pred[i] == 1:
                    t0_p1 = t0_p1 + 1
                elif y[i] == 0 and pred[i] == 0:
                    t0_p0 = t0_p0 + 1

        total_loss/= len(trainloader.dataset) 
        print('Total loss: {:.4f}'.format(total_loss))                                  ## Print model status
        print('t1_p1: ',t1_p1 , 't0_p1: ',t0_p1 )
        print('t1_p0: ',t1_p0 , 't0_p0: ',t0_p0 )
        return total_loss  

    def val(self,model,name,valloader):                                                 ## Test the model
        model.eval()                                                                    ## Set to evaluation mode
        val_loss = 0                                                                    ## Calculate total loss  
        t1_p1 = 0                                                                       ## Confusion matrix initialization
        t1_p0 = 0
        t0_p1 = 0
        t0_p0 = 0
        for x, y in valloader:
            x, y = Variable(x, volatile=True).cuda(), Variable(y,volatile=True).cuda()
            output = model(x)                                                           ## Use present model to forecast the the result 
            val_loss += F.binary_cross_entropy(output, y, size_average=False).data[0]   ## Sum up batch loss

            pred = np.empty([len(x),1])                                                 ## Calculate confusion matrix  
            output = output.cpu().data.numpy()
            for i in range(len(x)):  
                if(output[i] > self.threshold):
                    pred[i,0] = 1
                else:
                    pred[i,0] = 0
            y = y.cpu().data.numpy()
            
            for i in range(pred.shape[0]): 
                if y[i] == 1 and pred[i] == 1:
                    t1_p1 = t1_p1 + 1
                elif y[i] == 1 and pred[i] == 0:
                    t1_p0 = t1_p0 + 1
                elif y[i] == 0 and pred[i] == 1:
                    t0_p1 = t0_p1 + 1
                elif y[i] == 0 and pred[i] == 0:
                    t0_p0 = t0_p0 + 1

        val_loss /= len(valloader.dataset)
        print(name , ' set: Average loss: {:.4f}'.format(val_loss))                     ## Print model status
        print('t1_p1: ',t1_p1 , 't0_p1: ',t0_p1 )
        print('t1_p0: ',t1_p0 , 't0_p0: ',t0_p0 )
        return val_loss
 

class DNN(nn.Module):                                                                   ## Set up DNN
    def __init__(self,args):
        super(DNN, self).__init__()
        print(args.unit)
        self.den=nn.ModuleList()  
        for i in range(1,len(args.unit)-1):                                             ## Set up hidden layers
            self.den.append( nn.Sequential(
                nn.Linear(args.unit[i-1], args.unit[i]),                                ## Source: https://pytorch.org/docs/stable/nn.html
                nn.ReLU(),
                nn.Dropout(0.2)
            )) 
        self.den.append( nn.Sequential(
            nn.Linear(args.unit[-2], args.unit[-1]),
            nn.Dropout(0.2),
            nn.Sigmoid(),
        )) 

    def forward(self, x):                                                               ## Connect layers and activation function
        for i in self.den:
            x = i(x) 
        return x 