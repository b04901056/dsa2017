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

parser =  argparse.ArgumentParser(description='5_2 model')
parser.add_argument('-b',type=int,dest='batch_size',required=True)
parser.add_argument('-e',type=int,dest='epoch',required=True)
parser.add_argument('-trn',type=int,dest='train_num',required=True)
parser.add_argument('-exp',type=str,dest='exp',required=True)
args = parser.parse_args()
writer = SummaryWriter('runs/exp_'+args.exp)

class GRU (nn.Module):
    def  __init__(self, input_size, hidden_size=512, n_layers=2, dropout=0.5):
        super(GRU, self).__init__()
        self.hidden_size =  hidden_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=False)
        self.bn_0 = nn.BatchNorm1d(self.hidden_size)
        self.fc_1 = nn.Linear(self.hidden_size, int(self.hidden_size/2))
        self.bn_1 = nn.BatchNorm1d(int(self.hidden_size/2))
        self.fc_2 = nn.Linear(int(self.hidden_size), 11)
        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU()
    def forward(self, padded_sequence, input_lengths, hidden=None):
        self.lstm.flatten_parameters() 
        packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_lengths)
        outputs, (hn,cn) = self.lstm(packed, hidden)  
        outputs = self.bn_0(hn[-1])
        outputs = self.softmax(self.fc_2(outputs))
        return outputs

#train_features = np.load('5_1_data/video_data_train.npy')
valid_features = np.load('5_1_data/video_data_val.npy')
#train_y = np.load('5_1_data/video_label_train.npy')
valid_y = np.load('5_1_data/video_label_val.npy')
#print('train_features:',train_features[0].shape)
print('valid_features:',valid_features.shape)
#print('train_y:',train_y.shape)
print('valid_y:',valid_y.shape)


def single_batch_padding(train_X_batch, train_y_batch, test = False):
    if test==True:
        train_X_batch = [torch.FloatTensor(x) for x in train_X_batch]
        padded_sequence = nn.utils.rnn.pad_sequence(train_X_batch)
        label = torch.LongTensor(train_y_batch)
        length = [len(train_X_batch[0])]
    else:
        length = [len(x) for x in train_X_batch]
        #print('length: \n',length)
        #input()
        perm_index = np.argsort(length)[::-1]

        # sort by sequence length
        train_X_batch = [torch.FloatTensor(train_X_batch[i]) for i in perm_index]
        length = [len(x) for x in train_X_batch]
        #print('length: \n',length)
        #input()
        #print('train_X_batch[0]: \n',train_X_batch[0])
        #input()
        padded_sequence = nn.utils.rnn.pad_sequence(train_X_batch)
        label = torch.LongTensor(np.array(train_y_batch)[perm_index])
    return padded_sequence, label, length


feature_size = 1000
model = rnn = GRU(feature_size,hidden_size=512).cuda()
model.load_state_dict(torch.load('model/5-2/model_40.pt'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
BATCH_SIZE = 16
loss_function = nn.CrossEntropyLoss()
max_accuracy = 0
model.train()
'''
for epoch in range(100):
    print("\nEpoch:", epoch+1)
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
        if index+BATCH_SIZE > total_length: break
            
        # zero the parameter gradients
        optimizer.zero_grad()
        input_X = train_X_sfl[index:index+BATCH_SIZE]
        input_y = train_y_sfl[index:index+BATCH_SIZE] 

        # pad the sequence
        input_X, input_y, length = single_batch_padding(input_X, input_y)
        #print(input_X.size())
        #input()
        # use GPU
        input_X = input_X.cuda()
        # forward + backward + optimize
        output = model(input_X, length)
        loss = loss_function(output, input_y.cuda())
        loss.backward()
        optimizer.step()
        CE_loss += loss.cpu().data.numpy()
    print("training loss",CE_loss) 
    same_difference = []
    with torch.no_grad():
        model.eval()
        for i in range(len(valid_y)):
            input_valid_X, input_valid_y, valid_lengths = single_batch_padding([valid_features[i]], 
                                                                               [valid_y[i]],
                                                                               test=True)
            output = model(input_valid_X.cuda(),valid_lengths)
            output_label = torch.argmax(output,1).cpu().data
            #print(output_label)
            #print(input_valid_y)
            #print((output_label == input_valid_y).numpy())
            #input() 
            same_difference.append((output_label == input_valid_y).numpy())
        accuracy = np.mean(same_difference)
        print("validation accuracy: ",accuracy) 
        
    torch.save(model.state_dict(), 'model/5-2/model_'+str(epoch)+'.pt')
    model.train()

'''
# tsne
tsne_data = []
model.eval()
with torch.no_grad():
    for i in range(len(valid_y)):
        input_valid_X, input_valid_y, valid_lengths = single_batch_padding([valid_features[i]], 
                                                                           [valid_y[i]],
                                                                           test=True)
        output = model(input_valid_X.cuda(),valid_lengths)
        #output_label = torch.argmax(output,1).cpu().data  
        tsne_data.append(output.cpu().data)

tsne_data = np.concatenate(tsne_data,axis=0) 
print(tsne_data.shape)
print(valid_y)
#input()
X_embedded = TSNE(n_components=2,n_iter=20000,init='pca').fit_transform(tsne_data)
print(X_embedded.shape)
#input()

# plot
'''
plt.figure(figsize=(6, 5))
plt.scatter(X_embedded[:,0], X_embedded[:,1] , c=valid_y , cmap = plt.get_cmap('tab20'))
plt.show()
'''
x=X_embedded[:,0]
y=X_embedded[:,1]
classes = valid_y
unique = list(set(classes))
colors = [plt.cm.jet(float(i)/max(unique)) for i in unique]
for i, u in enumerate(unique):
    xi = [x[j] for j  in range(len(x)) if classes[j] == u]
    yi = [y[j] for j  in range(len(x)) if classes[j] == u]
    plt.scatter(xi, yi, c=colors[i], label='label '+str(u))
plt.legend(loc=2)

plt.show()
























































































































































