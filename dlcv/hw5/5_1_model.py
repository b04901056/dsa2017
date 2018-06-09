import scipy.misc
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import os
from torch.autograd import Variable
import torch.nn as nn
import argparse 
import torch.optim as optim
import torchvision
import random , time , math
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter 

parser =  argparse.ArgumentParser(description='5_1 model')
parser.add_argument('-b',type=int,dest='batch_size',required=True)
parser.add_argument('-e',type=int,dest='epoch',required=True)
parser.add_argument('-trn',type=int,dest='train_num',required=True)
parser.add_argument('-exp',type=str,dest='exp',required=True)
args = parser.parse_args()
writer = SummaryWriter('runs/exp_'+args.exp)

class imagedataset(Dataset):
    def __init__(self,img,label):
        self.img = img
        self.label = label 
    def __getitem__(self,i):
        x = self.img[i]
        y = self.label[i]
        return x,y 
    def __len__(self):
        return self.img.shape[0]

def get_data(img,label,batch_size,shuffle=False):
    img = np.load(img)
    label = np.load(label)
    data = []
    for x in img:
        half = int(x.shape[0]/2)
        last = x.shape[0]-1
        a = x[0].reshape(1,-1)
        b = x[half].reshape(1,-1)
        c = x[last].reshape(1,-1)
        data.append(np.concatenate((a,b,c),axis=1))
    data = np.array(data)
    print('data.shape:',data.shape)
    #input()
    data = torch.FloatTensor(data)
    label = torch.FloatTensor(label)
        
    dataset = imagedataset(data,label)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)

class fcn(nn.Module):
    def __init__(self):
        super(fcn,self).__init__()
        self.dnn1  = nn.Linear(3000,1024)
        self.dnn2  = nn.Linear(1024,256)
        self.dnn3  = nn.Linear(256,128)
        self.dnn4  = nn.Linear(128,64)
        self.dnn5  = nn.Linear(64,11)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self,x):
        h1 = self.dropout(self.dnn1(x))
        h2 = self.dnn2(h1)
        h3 = self.dropout(self.dnn3(h2))
        h4 = self.dnn4(h3) 
        h5 = self.dropout(self.dnn5(h4))
        h6 = self.softmax(h5)
        return h6

model = fcn().cuda()
print(model)
print('-'*50) 
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))
criterion = torch.nn.CrossEntropyLoss()

#training_set = get_data(img='5_1_data/video_data_train.npy',label='5_1_data/video_label_train.npy'\
                        #,batch_size=args.batch_size,shuffle=True)
validation_set = get_data(img='5_1_data/video_data_val.npy',label='5_1_data/video_label_val.npy'\
                        ,batch_size=args.batch_size,shuffle=False)
print('saved data ...')
print('start training ...')
#input()

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train_iter(epoch,model,iteration): 
    model.train()
    start = time.time()
    loss = 0
    accuracy = 0
    for step , (batch_x,batch_y) in enumerate(training_set):
        batch_idx = step + 1 
        batch_x = Variable(batch_x.squeeze(1)).cuda() 
        batch_size = len(batch_x) 

        label = Variable(batch_y.long(),volatile=True).cuda()  
        optimizer.zero_grad()  
        output = model(batch_x) 
        train_loss = criterion(output,label)
        train_loss.backward()
        optimizer.step()
        output = output.data.cpu().numpy()
        answer = np.argmax(output,axis=1) 

        count = 0
        for i in range(batch_size):
            if answer[i] == batch_y[i]: count += 1
        accu = count/batch_size 

        loss += train_loss.data[0]
        accuracy += accu
        writer.add_scalar('Training loss', train_loss.data[0] , iteration)
        writer.add_scalar('Training accuracy', accu , iteration)
        '''
        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)] | Accu: {:.6f} | | Loss: {:.6f} | step: {} | Time: {} '.format(
                   epoch 
                   , batch_idx * batch_size 
                   , len(training_set.dataset) 
                   , 100. * batch_idx * batch_size / len(training_set.dataset)
                   , accu
                   , train_loss.data[0]
                   , iteration
                   , timeSince(start, batch_idx* batch_size / len(training_set.dataset)))
                   , end='')
        '''
        iteration += 1 
     
    print('\n====> Training Epoch : {} | Time: {} | loss: {:.4f} | Accu: {:.4f} '.format(
                epoch 
                , timeSince(start,1)
                , loss / len(training_set)
                , accuracy / len(training_set) ))
    return iteration

def val(epoch,model,iteration): 
    model.eval()
    start = time.time()
    loss = 0
    accuracy = 0
    for step , (batch_x,batch_y) in enumerate(validation_set):
        batch_idx = step + 1 
        batch_x = Variable(batch_x.squeeze(1)).cuda() 
        batch_size = len(batch_x) 
        
        label = Variable(batch_y.long(),volatile=True).cuda()  
        #optimizer.zero_grad()  
        output = model(batch_x) 
        val_loss = criterion(output,label)
        #train_loss.backward()
        #optimizer.step()
        output = output.data.cpu().numpy()
        answer = np.argmax(output,axis=1) 

        count = 0
        for i in range(batch_size):
            if answer[i] == batch_y[i]: count += 1
        accu = count/batch_size 

        loss += val_loss.data[0]
        accuracy += accu
        writer.add_scalar('Validation loss', val_loss.data[0] , iteration)
        writer.add_scalar('Validation accuracy', accu , iteration)
        '''
        print('\rValidation Epoch: {} [{}/{} ({:.0f}%)] | Accu: {:.6f} | | Loss: {:.6f} | step: {} | Time: {} '.format(
                   epoch 
                   , batch_idx * batch_size 
                   , len(validation_set.dataset) 
                   , 100. * batch_idx * batch_size / len(validation_set.dataset)
                   , accu
                   , val_loss.data[0]
                   , iteration
                   , timeSince(start, batch_idx* batch_size / len(validation_set.dataset)))
                   , end='')
        '''
        iteration += 1
        
    print('\n====> Validation Epoch : {} | Time: {} | loss: {:.4f} | Accu: {:.4f} \n'.format(
                epoch 
                , timeSince(start,1)
                , loss / len(validation_set)
                , accuracy / len(validation_set) ))
    return iteration
'''
for i in range(args.epoch):
    epoch = i+1
    train_iter(epoch,model,(epoch-1)*len(training_set))
    val(epoch,model,(epoch-1)*len(validation_set))
    if epoch % 30 == 0:
        torch.save(model,'model/5-1/model_'+str(epoch)+'.pt')
'''
 

model = torch.load('model/5-1/model_990.pt')


'''
# tsne
tsne_data = []
valid_y = [] 
model.eval()
with torch.no_grad():
    for step , (batch_x,batch_y) in enumerate(validation_set):
            batch_idx = step + 1 
            batch_x = Variable(batch_x.squeeze(1)).cuda() 
            batch_size = len(batch_x) 
            
            label = Variable(batch_y.long(),volatile=True).cuda()  
            #optimizer.zero_grad()  
            output = model(batch_x) 
            tsne_data.append(output.cpu().data)
            valid_y.append(batch_y)

tsne_data = np.concatenate(tsne_data,axis=0) 
valid_y = np.concatenate(valid_y,axis=0).astype(int)
print(tsne_data.shape)
print(valid_y)
#input()
X_embedded = TSNE(n_components=2,n_iter=20000,init='pca').fit_transform(tsne_data)
print(X_embedded.shape)
#input()
'''

# plot
'''
plt.figure(figsize=(6, 5))
plt.scatter(X_embedded[:,0], X_embedded[:,1] , c=valid_y , cmap = plt.get_cmap('tab20'))
plt.show()

x=X_embedded[:,0]
y=X_embedded[:,1]
classes = valid_y
unique = list(set(classes))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
for i, u in enumerate(unique):
    xi = [x[j] for j  in range(len(x)) if classes[j] == u]
    yi = [y[j] for j  in range(len(x)) if classes[j] == u]
    plt.scatter(xi, yi, c=colors[i], label='label '+str(u))
#plt.legend(loc=2)

plt.show()
'''






















































































































































