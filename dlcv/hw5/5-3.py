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

video_path_val = sys.argv[1]
label_path_val = sys.argv[3]

model = models.vgg16(pretrained=True)
model.eval()
model.cuda()

video_data = []
label_data = []

video_path = os.listdir(video_path_val)
video_path.sort()
#print(video_path)
#input()
count = 0
for x in video_path:
    dir_path = os.path.join(video_path_val,x)
    image = os.listdir(dir_path)
    image.sort()
    img = []
    for y in image:
        image_path = os.path.join(dir_path,y)
        print('\r{}'.format(count),end='')
        count += 1
        #input()
        im = Image.open(image_path)
        #print(im)
        #input()

        new_im = im.resize((224, 224), Image.BICUBIC)
        #print(new_im)
        #input()
        new_im = torchvision.transforms.ToTensor()(new_im)
        new_im = torch.unsqueeze(new_im,0) 
        with torch.no_grad():
            new_im = Variable(new_im).cuda() 
            c = model(new_im)
        #print(c.size())
        #input()
        c = c.data.cpu().numpy()
        torch.cuda.empty_cache()
        img.append(c)
    img = np.concatenate(img,axis=0) 
    #print(img.shape)
    #input()
    video_data.append(img)

label_path = os.listdir(label_path_val)
label_path.sort()
#print(label_path)
#input()

for x in label_path:
    label = []
    path = os.path.join(label_path_val,x)
    print(path)
    #input()
    with open(path,'r') as f:
        while(True):
            a = f.readline().replace('\n','')
            if a == '' : break
            a = int(a)
            #print(a)
            #input()
            label.append(a)
    label = np.array(label).astype(int)
    #rint(label)
    #print(label.shape)
    #input()
    label_data.append(label) 

video_data = np.array(video_data)
label_data = np.array(label_data) 


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
    def forward(self, sequence, hidden=None):
        #packed = torch.nn.utils.rnn.pack_padded_sequence(padded_sequence, input_lengths)
        self.lstm.flatten_parameters() 
        outputs, (hn,cn) = self.lstm(sequence, hidden)  
        #print(outputs.size())
        #input()
        output = []
        for i in range(sequence.size()[0]):
            x = self.bn_0(outputs[i])
            x = self.softmax(self.fc_2(x)) #(batch_size,11)
            output.append(x)
        output = torch.cat(output,0) 
        return output

#valid_features = video_data
#valid_y = label_data

valid_features = np.load('5_3_data/video_data_val.npy')  
valid_y = np.load('5_3_data/label_data_val.npy')

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
model = GRU(feature_size,hidden_size=512).cuda() 
model.load_state_dict(torch.load('model/5-3_new/model_100.pt')) 
 
model.eval() 
path = sys.argv[2]
for i in range(len(valid_y)):
    number = 0
    answer = [] 
    with torch.no_grad(): 
        input_valid_X, input_valid_y = single_batch_padding([valid_features[i]], 
                                                            [valid_y[i]],
                                                            test=True)
        output = model(input_valid_X.cuda()) 
        number += len(valid_features[i])
        output_label = torch.argmax(output,1).cpu().data   
        answer.append(output_label)

    answer = np.concatenate(answer,axis=0).reshape(-1)
    #print(answer.shape)
    with open(path+'/p3_valid_'+str(i)+'.txt','w') as f:
        for x in range(number):
            f.write(str(answer[x]))
            f.write('\n')
    answer_ = valid_y[i] 
    #print(answer_.shape)
    with open('p3_valid_answer_'+str(i)+'.txt','w') as f:
        for x in range(number):
            f.write(str(answer[x]))
            f.write(' ')
            f.write(str(answer_[x]))
            f.write('\n')
    comp = np.array(answer==answer_)
    print('accu_'+str(i)+' = ',np.mean(comp))



















































































































































