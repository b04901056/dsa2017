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
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensorboardX import SummaryWriter 

parser =  argparse.ArgumentParser(description='5_3 model')
parser.add_argument('-b',type=int,dest='batch_size',required=True)
parser.add_argument('-e',type=int,dest='epoch',required=True)
parser.add_argument('-trn',type=int,dest='train_num',required=True)
parser.add_argument('-exp',type=str,dest='exp',required=True)
args = parser.parse_args()
writer = SummaryWriter('runs/exp_'+args.exp)

class RNN (nn.Module):
    def  __init__(self, input_size, hidden_size=512, n_layers=2, dropout=0.5):
        super(RNN, self).__init__()
        self.hidden_size =  hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=False)
        self.bn_0 = nn.BatchNorm1d(self.hidden_size)
        self.fc_1 = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.bn_1 = nn.BatchNorm1d(int(self.hidden_size/2))
        self.fc_2 = nn.Linear(int(self.hidden_size), 11)
        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU()
    def forward(self, sequence, hidden=None):
        #packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_lengths)
        #self.lstm.flatten_parameters() 
        outputs, (hn,cn) = self.lstm(sequence, hidden)  
        #print(outputs.size())
        output = []
        for i in range(sequence.size()[0]):
            #x = self.bn_0(outputs[i])
            x = outputs[i]
            #x = self.fc_1(x)
            x = self.softmax(self.fc_2(x)) #(batch_size,11)
            output.append(x)
        output = torch.cat(output,0) 
        return output

train_features = np.load('5_3_data/video_data_train.npy')
valid_features = np.load('5_3_data/video_data_val.npy')
train_y = np.load('5_3_data/label_data_train.npy')
valid_y = np.load('5_3_data/label_data_val.npy')
#print('train_features:',train_features)
#input()
#print('valid_features:',valid_features)
#input()
#print('train_y:',train_y)
#input()
#print('valid_y:',valid_y)  
#input()
#print('train_features[0]:',train_features[0].shape) 
#input()


max_step = 512
def single_batch_padding(train_X_batch, train_y_batch, test = False):
    if test == True:
        train_X = torch.FloatTensor(np.array(train_X_batch)) 
        label = torch.LongTensor(train_y_batch) 
    else:
        train_X = []
        train_Y = []
        for i in range(len(train_X_batch)):
            if len(train_X_batch[i]) > max_step:
                rand = random.sample(range(len(train_X_batch[i])), max_step)
                rand.sort()
                #print(rand)
                #input() 
                selected_x = train_X_batch[i][rand]
                selected_y = train_y_batch[i][rand]
                #print(selected_x.shape)
                #input()
                train_X.append(selected_x)
                train_Y.append(selected_y)
            else: 
                train_X.append(train_X_batch[i])
                train_Y.append(train_y_batch[i])
    
        train_X = torch.FloatTensor(np.array(train_X)) 
        #print('train_X[0]: \n',train_X[0])
        #input()
        #padded_sequence = nn.utils.rnn.pad_sequence(train_X)
        label = torch.LongTensor(np.array(train_Y))
    return train_X, label 


feature_size = 1000
model = RNN(feature_size,hidden_size=512).cuda()
model.load_state_dict(torch.load('model/5-2/model_40.pt'))
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
BATCH_SIZE = 8
loss_function = nn.CrossEntropyLoss()
max_accuracy = 0 

model.train()
iteration_loss = 0
iteration_accu = 0
for epoch in range(5000):
    
    print("\nEpoch:", epoch+1)
    #start = time.time()
    CE_loss = 0.0
    total_length = len(train_features)
    #print('total_length = ',total_length) 
    #input()
    # shuffle
    perm_index = np.random.permutation(len(train_features))
    train_X_sfl = [ train_features[i] for i in perm_index]
    train_y_sfl = np.array(train_y)[perm_index]
    # construct training batch
    for index in range(0,total_length ,BATCH_SIZE):
        #print('\rindex = {}'.format(index+BATCH_SIZE),end='')
        if index+BATCH_SIZE > total_length: break
            
        # zero the parameter gradients
        optimizer.zero_grad()
        input_X = train_X_sfl[index:index+BATCH_SIZE]
        input_y = train_y_sfl[index:index+BATCH_SIZE] 

        # pad the sequence
        input_X, input_y = single_batch_padding(input_X, input_y)
        input_X = input_X.permute(1,0,2)
        input_y = input_y.permute(1,0)
        #print(input_X.size())
        #print(input_y.size())
        #input() 
        # use GPU 
        input_X = input_X.cuda() 
        # forward + backward + optimize
        output = model(input_X) 
        total_loss = 0 
        #print( (index+BATCH_SIZE) / total_length)
        #print('\rTime: {} '.format( timeSince(start, (index+BATCH_SIZE) / total_length)))
        for i in range(input_X.size()[0]): 
            #print(output[BATCH_SIZE*i:BATCH_SIZE*(i+1)])
            #input()
            #print(output[i])
            #input()
            #print(input_y[i])
            #input()
            loss = loss_function(output[BATCH_SIZE*i:BATCH_SIZE*(i+1)], input_y[i].cuda()) 
            #loss = loss_function(output[i], input_y[i].cuda())
            #loss.backward(retain_graph=True)
            total_loss += loss
        total_loss /= input_X.size()[0]
        total_loss.backward()
        writer.add_scalar('CrossEntropyLoss', total_loss.item() , iteration_loss)
        iteration_loss += 1 
        optimizer.step()
        
        CE_loss += total_loss.cpu().data.numpy()
    print("training loss",CE_loss) 
    
    same_difference = [] 
    
    with torch.no_grad():
        model.eval()
        label = []
        target = []
        for i in range(len(valid_y)):
            input_valid_X, input_valid_y = single_batch_padding( [valid_features[i]]
                                                                , [valid_y[i]]      
                                                                , test=True )
            output = model(input_valid_X.cuda())
            output_label = torch.argmax(output,1).cpu().data 
            output_label = output_label.view(output_label.size()[0],1)
            input_valid_y = input_valid_y.permute(1,0)
            #print('output_label',output_label.size())
            #print('input_valid_y',input_valid_y.size())
            #input()
            ac = (output_label == input_valid_y).numpy()
            print(np.mean(ac))
            label.append(output_label)
            target.append(input_valid_y)
        '''
        acc = 0
        for i in range(5):
            acc += np.mean(same_difference[i])   
        accuracy = acc / 5
        '''
        label = torch.cat(label,0).numpy()
        target = torch.cat(target,0).numpy()
        #print(len(label))
        #print(len(target))
        #input()
        accuracy = (label == target)
        #print(accuracy)
        #input()
        accuracy = np.mean(accuracy)
        print("validation accuracy: ",accuracy)
        writer.add_scalar('Accuracy', accuracy , iteration_accu)
        iteration_accu += 1

    if epoch % 50 == 0 :
        torch.save(model.state_dict(), 'model/5-3_new/model_'+str(epoch)+'.pt')
    model.train()

'''
## test accuracy 
feature_size = 1000
model = GRU(feature_size,hidden_size=512).cuda() 
model.load_state_dict(torch.load('model/5-3/model_9800.pt')) 
same_difference = [] 
with torch.no_grad():
    model.eval()
    for i in range(len(valid_y)):
        input_valid_X, input_valid_y = single_batch_padding( [valid_features[i]]
                                                            , [valid_y[i]]      
                                                            , test=True )
        output = model(input_valid_X.cuda())
        output_label = torch.argmax(output,1).cpu().data 
        #print('output_label',output_label)
        #print('input_valid_y',input_valid_y)
        #input()
        same_difference.append((output_label == input_valid_y).numpy()) 
    acc = 0
    for i in range(5):
        print(np.mean(same_difference[i]))
        acc += np.mean(same_difference[i])   
    accuracy = acc / 5
    print("validation accuracy: ",accuracy)  
'''
'''
select = 4
feature_size = 1000
model = GRU(feature_size,hidden_size=512).cuda() 
model.load_state_dict(torch.load('model/5-3/model_9800.pt')) 
same_difference = [] 
with torch.no_grad():
    model.eval() 
    input_valid_X, input_valid_y = single_batch_padding( [valid_features[select]]
                                                        , [valid_y[select]]      
                                                        , test=True )
    output = model(input_valid_X.cuda())
    output_label = torch.argmax(output,1).cpu().data 
# color bar 
color = {0:[255,255,255]
        ,1:[255,0,0]
        ,2:[0,255,0]
        ,3:[0,0,255]
        ,4:[0,255,255]
        ,5:[255,0,255]
        ,6:[0,0,0]
        ,7:[255,255,0]
        ,8:[128,0,0]
        ,9:[0,128,0]
        ,10:[0,0,128]}

start = 690
end = valid_y[select].shape[0]-1
fig = plt.figure(figsize=(8, 3)) 
img = [i for i in range(1,16310,12)]
img = img[690:]
a = len(img)/10
for i in range(10):
    print(img[int(a*i)],' ',end='')

label = valid_y[select].tolist()
#print(output_label)
#input()
color_list = []
for i in range(len(label)):
    color_list.append(color[label[i]])
color_list = np.array(color_list)[start:end]/255 
cmap = mpl.colors.ListedColormap(color_list)
cmap.set_over('0.25')
cmap.set_under('0.75') 
bounds = [i for i in range(start,end)] 
ax1 = fig.add_axes([0.05, 0.775, 0.9, 0.15])
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb2 = mpl.colorbar.ColorbarBase(ax=ax1,
                                cmap=cmap,
                                norm=norm, 
                                boundaries=bounds, 
                                spacing='proportional',
                                orientation='horizontal')

output_label = output_label.numpy().astype(int).tolist() 
color_list = []
for i in range(len(output_label)):
    color_list.append(color[output_label[i]])
color_list = np.array(color_list)[start:end]/255
#for i in range(color_list.shape[0]):
    #print(color_list[i])
#input()
cmap = mpl.colors.ListedColormap(color_list)
cmap.set_over('0.25')
cmap.set_under('0.75') 
bounds = [i for i in range(start,end)]
#bounds = [1, 2, 4, 7, 8,9]
ax2 = fig.add_axes([0.05, 0.15, 0.9, 0.15])
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
cb2 = mpl.colorbar.ColorbarBase(ax=ax2,
                                cmap=cmap,
                                norm=norm, 
                                boundaries=bounds, 
                                spacing='proportional',
                                orientation='horizontal')
plt.show()
'''






















































































































































