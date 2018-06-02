import numpy as np 
import reader
from torchvision import models
import torchvision
from torch.autograd import Variable
from PIL import Image
import sys
import os
import torch

video_path_train = 'data/FullLengthVideos/videos/train'
video_path_val = 'data/FullLengthVideos/videos/valid'
label_path_train = 'data/FullLengthVideos/labels/train'
label_path_val = 'data/FullLengthVideos/labels/valid'

model = models.vgg16(pretrained=True)
model.eval()
model.cuda()

video_data = []
label_data = []

video_path = os.listdir(video_path_val)
video_path.sort()
#print(video_path)
#input()

for x in video_path:
    dir_path = os.path.join(video_path_val,x)
    image = os.listdir(dir_path)
    image.sort()
    img = []
    for y in image:
        image_path = os.path.join(dir_path,y)
        print(image_path)
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
print(video_data.shape)
print(label_data.shape)
input()
np.save('video_data_val.npy',video_data)
np.save('label_data_val.npy',label_data)
print('data saved ...')

