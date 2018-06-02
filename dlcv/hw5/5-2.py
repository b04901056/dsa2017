import scipy.misc
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import os
import reader
from torch.autograd import Variable
import torch.nn as nn
import argparse 
import torch.optim as optim
import torchvision
import random , time , math
from sklearn.manifold import TSNE
import sys
from torchvision import models
from matplotlib import pyplot as plt 
'''
video_path_val = sys.argv[1]

data = reader.getVideoList(sys.argv[2])
dic = {}
name = data['Video_name']
label = data['Action_labels']
#print('label:',label)
#input()
for i in range(len(label)): 
    dic[name[i]] = label[i] 

model = models.vgg16(pretrained=True)
model.eval()
model.cuda()

video_data = []
video_label = []

video_path = os.listdir(video_path_val)
video_path.sort()
count = 0 
for cat in video_path:
    path = os.path.join(video_path_val,cat)
    video_path_in = os.listdir(path)
    video_path_in.sort()
    for x in video_path_in:
        video_label.append(dic[x[:-20]]) 
        print('\r{}'.format(count),end='')
        count += 1
        #input()
        a = reader.readShortVideo(video_path_val,cat,x)
        #print(a.shape)
        #input()
        #print(model)
        #input() 
        img = []
        for i in range(a.shape[0]):
            b = a[i]
            b = Image.fromarray(b)
            #print(b)
            #input()
            new_im = b.resize((224, 224), Image.BICUBIC)
            #print(new_im)
            #input()
            new_im = torchvision.transforms.ToTensor()(new_im)
            new_im = torch.unsqueeze(new_im,0)
            #img.append(torch.unsqueeze(new_im,0))
            #img = torch.cat(img,0)
            with torch.no_grad():
                new_im = Variable(new_im).cuda()
            #print(img)
            #input()
            c = model(new_im)
            #print(c)
            #input()
            c = c.data.cpu().numpy()
            torch.cuda.empty_cache()
            img.append(c)
        img = np.concatenate(img,axis=0)
        #print(img)
        #print(img.shape)
        #input()
        video_data.append(img)
            
video_data = np.array(video_data)
video_label = np.array(video_label).astype(int) 
 
'''
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

#valid_features = video_data
#valid_y = video_label

valid_features = np.load('5_1_data/video_data_val.npy') 
valid_y = np.load('5_1_data/video_label_val.npy')

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
 
answer = [] 
model.eval()
with torch.no_grad():
    for i in range(len(valid_y)):
        input_valid_X, input_valid_y, valid_lengths = single_batch_padding([valid_features[i]], 
                                                                           [valid_y[i]],
                                                                           test=True)
        output = model(input_valid_X.cuda(),valid_lengths)
        output_label = torch.argmax(output,1).cpu().data  
        answer.append(output_label)

answer = np.concatenate(answer,axis=0).reshape(-1) 
with open('p2_valid.txt','w') as f:
    for i in range(517):
        f.write(str(answer[i]))
        f.write('\n')

answer_ = np.load('5_1_data/video_label_val.npy') 
with open('p2_valid_answer.txt','w') as f:
    for i in range(517):
        f.write(str(answer[i]))
        f.write(' ')
        f.write(str(answer_[i]))
        f.write('\n')

comp = np.array(answer==answer_)
print('accu = ',np.mean(comp))



















































































































































