import scipy.misc
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import os
import sys
from torchvision import models
import reader
from torch.autograd import Variable
import torch.nn as nn
import argparse 
import torch.optim as optim
import torchvision
import random , time , math
from sklearn.manifold import TSNE
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
video_data = np.load('5_1_data/video_data_val.npy') 
video_label = np.load('5_1_data/video_label_val.npy')

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

def get_data(batch_size,shuffle=False):
    img = video_data
    label = video_label
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

validation_set = get_data(batch_size=16,shuffle=False)


model = torch.load('model/5-1/model_990.pt')
model.eval()
answer = []
with torch.no_grad():
    for step , (batch_x,batch_y) in enumerate(validation_set):
        batch_idx = step + 1 
        batch_x = Variable(batch_x.squeeze(1)).cuda() 
        batch_size = len(batch_x) 
        with torch.no_grad():
            label = batch_y.cuda()  
        #optimizer.zero_grad()  
        output = model(batch_x) 
        output = output.data.cpu().numpy()
        a = np.argmax(output,axis=1)  
        answer.append(a) 

answer = np.concatenate(answer,axis=0).reshape(-1,1)
print(answer.shape)
with open('p1_valid.txt','w') as f:
    for i in range(517):
        f.write(str(answer[i][0]))
        f.write('\n')

answer_ = np.load('5_1_data/video_label_val.npy') 
with open('p1_valid_answer.txt','w') as f:
    for i in range(517):
        f.write(str(answer[i][0]))
        f.write(' ')
        f.write(str(answer_[i]))
        f.write('\n')

















































































































































