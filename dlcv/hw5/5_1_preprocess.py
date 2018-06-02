import numpy as np 
import reader
from torchvision import models
import torchvision
from torch.autograd import Variable
from PIL import Image
import sys
import os
import torch

video_path_train = 'data/TrimmedVideos/video/train'
video_path_val = 'data/TrimmedVideos/video/valid'

data = reader.getVideoList('data/TrimmedVideos/label/gt_valid.csv')
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
for cat in video_path:
    path = os.path.join(video_path_val,cat)
    video_path_in = os.listdir(path)
    video_path_in.sort()
    for x in video_path_in:
        video_label.append(dic[x[:-20]])
        print(x[:-20])
        print(dic[x[:-20]])
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
print('video_label:',video_label)
input()
np.save('5_1_data/video_data_train.npy',video_data)
np.save('5_1_data/video_label_train.npy',video_label)
print('data saved ...')
'''
video_data = np.load('video_data.npy')
vidoe_label = np.load('vidoe_label.npy')
video_number = np.load('video_number.npy')
for i in range(len(video_data)):
	print(video_data[i].shape)
input()
print(video_data.shape)
print(vidoe_label.shape)
print(video_number.shape)
'''
