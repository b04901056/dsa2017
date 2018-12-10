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
from imblearn.combine import SMOTEENN, SMOTETomek 
from sklearn.decomposition import PCA 

class Datamanager():
    def __init__(self):
        self.dataset = {}
        self.Sequence = {} ## check the category of attribute #5, and assing an integer to each of it 
        self.threshold = 1e-10
        self.over_sample = False
        self.over_sample_rate = 150
        self.down_sample = False
        self.down_sample_rate = 2
        self.synthesis = True
        self.synthesis_rate = 150
        self.smote = False 
        self.weighted_loss = True
        self.weighted_loss_rate = 0.1 

        self.pca = PCA(n_components = 128)

    def get_data(self,name,file_name,b_size,args,shuf=True):     
        with open(file_name,newline='') as csvfile:
            rows = csv.reader(csvfile)
            data = []
            for row in rows:
                data.append(row) 
            data = data[2:]
            
            data = np.array(data)
            if name == 'train' :                    ## missing data
                data = np.delete(data,2,0)            
            if name == 'test' :  
                data = np.delete(data,1103,0) 
                data = np.delete(data,1102,0) 
                data = np.delete(data,699,0) 
                data = np.delete(data,5,0) 

            data = np.delete(data,176,1)            ## these columns have std = 0
            data = np.delete(data,167,1)
            data = np.delete(data,166,1)
            data = np.delete(data,165,1)         
            data = np.delete(data,5,1)
            data = np.delete(data,4,1)

            if name == 'train' :
                count = -4
                for i in range(data.shape[0]):
                    if data[i][4] not in self.Sequence:
                        self.Sequence[data[i][4]] = count
                        count = count + 1 
                #for key in self.Sequence.keys():
                    #print(key,': ',self.Sequence[key])
            '''
            for j in range(data.shape[0]):
                if data[j][4] in self.Sequence.keys(): 
                    data[j][4] = self.Sequence[data[j][4]]
                else :
                    data[j][4] = random.random()
            '''
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

            if name == 'train' :
                np.random.shuffle(data) 

                Y = data[:int(data.shape[0]*0.9),3].reshape(-1,1).astype(np.double)
                Y_val = data[int(data.shape[0]*0.9):,3].reshape(-1,1).astype(np.double) 
 
                self.pca.fit(np.delete(data,3,1))
                X = self.pca.transform(np.delete(data,3,1).astype(np.double)[:int(data.shape[0]*0.9),:])
                X_val = self.pca.transform(np.delete(data,3,1).astype(np.double)[int(data.shape[0]*0.9):,:])

                if self.over_sample or self.down_sample or self.synthesis or self.smote:
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
                    if self.over_sample:
                        add_one_X , add_one_Y = X[count_1_list] , Y[count_1_list]
                        #print(add_one_X)
                        #print(add_one_Y)
                        noise = np.random.normal(0, 0.3, add_one_X.shape)
                        add_one_X = add_one_X + noise 
                        for i in range(self.over_sample_rate):
                            X = np.concatenate((X,add_one_X),axis = 0)
                            Y = np.concatenate((Y,add_one_Y),axis = 0)

                    if self.down_sample:
                        number = int(count_0 - count_1 * (self.over_sample_rate + 1) * self.down_sample_rate)
                        while(number > 0): 
                            for i in range(Y.shape[0]):
                                if Y[i][0] == 0:
                                    X = np.delete(X,i,0)
                                    Y = np.delete(Y,i,0)
                                    number = number - 1
                                    break
                    if self.synthesis:
                        synthesis_number = count_1 * self.synthesis_rate
                        for i in range(synthesis_number): 
                            add_one_X , add_one_Y = np.empty([1,X.shape[1]]) , np.ones((1,1))
                            for i in range(add_one_X.shape[1]):
                                add_one_X[0][i] = X[random.choice(count_1_list)][i] 
                            X = np.concatenate((X,add_one_X),axis = 0)
                            Y = np.concatenate((Y,add_one_Y),axis = 0)
                    if self.smote: 
                        print(X.shape)
                        print(Y.shape) 
                        sm = SMOTEENN(sampling_strategy = 1)
                        X, Y = sm.fit_resample(X, Y)
                        Y = Y.reshape(-1,1) 
                        print(X.shape)
                        print(Y.shape) 
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

                
                X,Y=torch.from_numpy(X).cuda(),torch.from_numpy(Y).cuda()
                train_dataset = Data.TensorDataset(data_tensor=X[:], target_tensor=Y[:]) 
                self.dataset['train']=Data.DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=shuf)

                X_val,Y_val=torch.from_numpy(X_val).cuda(),torch.from_numpy(Y_val).cuda()
                val_dataset = Data.TensorDataset(data_tensor=X_val[:], target_tensor=Y_val[:]) 
                self.dataset['val']=Data.DataLoader(dataset=val_dataset, batch_size=b_size, shuffle=shuf) 
 
            elif name == 'test': 
                Y = data[:,3].reshape(-1,1).astype(np.double) 
                X = self.pca.transform(np.delete(data,3,1).astype(np.double))

                X,Y=torch.from_numpy(X).cuda(),torch.from_numpy(Y).cuda()
                train_dataset = Data.TensorDataset(data_tensor=X[:], target_tensor=Y[:]) 
                self.dataset['test']=Data.DataLoader(dataset=train_dataset, batch_size=b_size, shuffle=shuf)
    
    def train(self,model,trainloader,epoch): 
        model.train()
        optimizer = torch.optim.Adam(model.parameters())   # optimize all dnn parameters   
        loss_func = nn.BCELoss()
        total_loss = 0
        correct = 0
        t1_p1 = 0
        t1_p0 = 0
        t0_p1 = 0
        t0_p0 = 0

        for batch_index, (x, y) in enumerate(trainloader):
            x, y= Variable(x).cuda(), Variable(y).cuda() 
            output = model(x)
            if self.weighted_loss:
                weight = np.empty([len(x)])
                for i in range(len(x)):
                    weight[i] = self.weighted_loss_rate
                arr = np.where(y.data == 1)
                weight[arr[0].tolist()] = 1 - self.weighted_loss_rate 
                weight = torch.from_numpy(weight).cuda().double().view(len(x),1)
                #print(weight)
                #print(x.size())
                #print(y.size())
                loss_func = nn.BCELoss(weight = weight)
            loss = loss_func(output,y) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_index % 4 == 0:
                print('\rTrain Epoch: {} | [{}/{} ({:.0f}%)]\t '.format(
                        epoch, batch_index * len(x), len(trainloader.dataset),
                        100. * batch_index / len(trainloader)),end='')

            total_loss+= loss.data[0]*len(x) # sum up batch loss 

            pred = np.empty([len(x),1])      # calculate accuracy
            output = output.cpu().data.numpy()
            #print('output',output)
            for i in range(len(x)):  
                if(output[i] > self.threshold):
                    pred[i,0] = 1
                else:
                    pred[i,0] = 0

            y = y.cpu().data.numpy()
            #accu = (y == pred) 
            #print('y',y)
            #print('pred',pred) 
             
            for i in range(pred.shape[0]):
                if pred[i] == y[i]:
                    correct = correct + 1 
                if y[i] == 1 and pred[i] == 1:
                    t1_p1 = t1_p1 + 1
                elif y[i] == 1 and pred[i] == 0:
                    t1_p0 = t1_p0 + 1
                elif y[i] == 0 and pred[i] == 1:
                    t0_p1 = t0_p1 + 1
                elif y[i] == 0 and pred[i] == 0:
                    t0_p0 = t0_p0 + 1

        total_loss/= len(trainloader.dataset) 
        print('Total loss: {:.4f} , Accuracy: {}/{} ({:.0f}%)'.format(total_loss, correct, len(trainloader.dataset),
            100. * correct / len(trainloader.dataset)))
        print('t1_p1: ',t1_p1 , 't1_p0: ',t1_p0 , 't0_p1: ',t0_p1 , 't0_p0: ',t0_p0)
        return total_loss , 100 * correct / len(trainloader.dataset)

    def val(self,model,name,valloader):
        model.eval()
        val_loss = 0
        correct = 0
        t1_p1 = 0
        t1_p0 = 0
        t0_p1 = 0
        t0_p0 = 0
        for x, y in valloader:
            x, y = Variable(x, volatile=True).cuda(), Variable(y,volatile=True).cuda()
            output = model(x)
            val_loss += F.binary_cross_entropy(output, y, size_average=False).data[0] # sum up batch loss

            pred = np.empty([len(x),1])      # calculate accuracy
            output = output.cpu().data.numpy()
            for i in range(len(x)):  
                if(output[i] > self.threshold):
                    pred[i,0] = 1
                else:
                    pred[i,0] = 0
            y = y.cpu().data.numpy()
            accu = (y == pred)  
            for i in range(pred.shape[0]):
                if pred[i] == y[i]:
                    correct = correct + 1 
                if y[i] == 1 and pred[i] == 1:
                    t1_p1 = t1_p1 + 1
                elif y[i] == 1 and pred[i] == 0:
                    t1_p0 = t1_p0 + 1
                elif y[i] == 0 and pred[i] == 1:
                    t0_p1 = t0_p1 + 1
                elif y[i] == 0 and pred[i] == 0:
                    t0_p0 = t0_p0 + 1

        val_loss /= len(valloader.dataset)
        print(name , ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            val_loss, correct, len(valloader.dataset),
            100. * correct / len(valloader.dataset)))
        print('t1_p1: ',t1_p1 , 't1_p0: ',t1_p0 , 't0_p1: ',t0_p1 , 't0_p0: ',t0_p0)
        return val_loss , 100 * correct / len(valloader.dataset)
 

class DNN(nn.Module):
    def __init__(self,args):
        super(DNN, self).__init__()
        print(args.unit)
        self.den=nn.ModuleList() 
        
        for i in range(1,len(args.unit)-1):
            self.den.append( nn.Sequential(
                nn.Linear(args.unit[i-1], args.unit[i]),
                nn.ReLU(),
                nn.Dropout(0.5)
            ))

        self.den.append( nn.Sequential(
            nn.Linear(args.unit[-2], args.unit[-1]),
            nn.Dropout(0.2),
            nn.Sigmoid(),
        )) 

    def forward(self, x):
        for i in self.den:
            x = i(x) 
        return x 