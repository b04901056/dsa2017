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
import torchvision.transforms as transforms
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter 

parser =  argparse.ArgumentParser(description='vae model')
parser.add_argument('-tn',type=int,dest='train_num',required=True)
parser.add_argument('-b',type=int,dest='batch_size',required=True)
parser.add_argument('-lat',type=int,dest='latent_size',required=True)
parser.add_argument('-tm',type=int,dest='test_num',required=True)
parser.add_argument('-ra',type=int,dest='rand',required=True)

args = parser.parse_args()

class imagedataset(Dataset):
    def __init__(self,img,mode):
        self.img = img
        self.mode = mode 
    def __getitem__(self,i):
        if self.mode == 'train':
            x = self.img[i]
            y = self.img[i]
            return x,y,i
        elif self.mode == 'test':
            x = self.img[i]
            return x,i 
        else: return ValueError('Wrong mode')
    def __len__(self):
        return self.img.shape[0]

def get_data(path,num,name,batch_size,shuffle=False):
    data = os.listdir(path)
    data.sort()
    data = data[:num]
    arr = []
    for x in data:
        print(os.path.join(path,x))
        #input()
        im = Image.open(os.path.join(path,x))
        im = np.array(im)
        #print(im)
        #input()
        im = (im/127.5)-1
        #print(im)
        #input()
        #print(im)
        #input()
        arr.append(im)
    arr = np.array(arr)
    print('saving data ...')
    np.save(name+'.npy',arr)
    arr = torch.FloatTensor(arr)
    dataset = imagedataset(arr,name)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)

class vae(nn.Module):
    def __init__(self,latent_size,nc,ngf,ndf):
        super(vae,self).__init__()
        self.latent_size = latent_size
        self.nc = nc #3
        self.ngf = ngf #64
        self.ndf = ndf #64
        self.latent_size = latent_size
        
        #(3,64,64)
        self.e1 = nn.Conv2d(3,32,4,2,1) #(32,32,32)
        self.bn1 = nn.BatchNorm2d(32)

        self.e2 = nn.Conv2d(32,64,4,2,1) #(64,16,16)
        self.bn2 = nn.BatchNorm2d(64)
 
        self.e3 = nn.Conv2d(64,128,4,2,1) #(128,8,8)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.e4 = nn.Conv2d(128,256,4,2,1) #(256,4,4)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.e5 = nn.Conv2d(256,512,4,2,1) #(512,2,2)
        self.bn5 = nn.BatchNorm2d(512)

        self.e6 = nn.Conv2d(512,1024,4,2,1) #(1024,1,1)
        self.bn6 = nn.BatchNorm2d(1024)
 
        self.mu = nn.Linear(1024,latent_size)
        self.sigma = nn.Linear(1024,latent_size) # 512 

        self.d1 = nn.Linear(latent_size, 1024) # (1024,1,1)
        
        self.up1 = nn.ConvTranspose2d(1024,512,4,2,1) 
        self.bn7 = nn.BatchNorm2d(512, 1.e-3)

        self.up2 = nn.ConvTranspose2d(512,256,4,2,1)
        self.bn8 = nn.BatchNorm2d(256, 1.e-3)

        self.up3 = nn.ConvTranspose2d(256,128,4,2,1)
        self.bn9 = nn.BatchNorm2d(128, 1.e-3)

        self.up4 = nn.ConvTranspose2d(128,64,4,2,1)
        self.bn10 = nn.BatchNorm2d(64, 1.e-3)
 
        self.up5 = nn.ConvTranspose2d(64,64,4,2,1)
        self.bn11 = nn.BatchNorm2d(64, 1.e-3)
 
        self.up6 = nn.ConvTranspose2d(64,3,4,2,1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    def encode(self,x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        #print(h5.size())
        #input()
        h6 = self.leakyrelu(self.bn6(self.e6(h5)))
 
        h6 = h6.view(-1,1024)
        return self.mu(h6),self.sigma(h6)
    
    def reparametrize(self,mu,logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
   
    def decode(self,z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, 1024, 1, 1)
        #print('h1:',h1.size())
        h2 = self.leakyrelu(self.bn7(self.up1(h1)))
        #print('h2:',h2.size())
        h3 = self.leakyrelu(self.bn8(self.up2(h2)))
        #print('h3:',h3.size())
        h4 = self.leakyrelu(self.bn9(self.up3(h3)))
        #print('h4:',h4.size())
        h5 = self.leakyrelu(self.bn10(self.up4(h4)))
        #print('h5:',h5.size())
        h6 = self.leakyrelu(self.bn11(self.up5(h5)))
        #print('h6:',h6.size())
        h7 = self.tanh(self.up6(h6))
        #print('h7:',h7.size())
        return h7.permute(0,2,3,1)
        #return h7.view(-1,64,64,3)
        '''
    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z
        '''
    def forward(self,x):
        mu, logvar = self.encode(x.permute(0,3,1,2))
        #print('mu:',mu)
        #print('logvar:',logvar)
        z = self.reparametrize(mu, logvar)
        #print('z:',z)
        res = self.decode(z)
        #print('res:',res)
        #input()
        return res, mu, logvar 


model = vae(nc=3, ngf=64, ndf=64, latent_size = args.latent_size ).cuda()
#testing_set = get_data('test',num=2621,name='test',batch_size=args.batch_size,shuffle=False)
arr = np.load('test.npy')
arr = torch.FloatTensor(arr) 
dataset = imagedataset(arr,'test')
testing_set = DataLoader(dataset,batch_size=args.batch_size,shuffle=False)

#tsne
z = Variable(testing_set.dataset.img, volatile=True)
z = z.cuda()
model = torch.load('model/vae/model_300.pt')
recon = model.encode(z.permute(0,3,1,2))[0].data.cpu().numpy()[:1500]
#print('recon:',recon)
#print('recon size:',recon.shape)
#input()
test_label = []
label = []
with open('test.csv') as f:
    f.readline()
    for i in range(2621):
        attr = f.readline().replace('\n','').split(',')[1:]
        a = float(attr[7])
        test_label.append(a)
        
for i in range(len(test_label)):
    if test_label[i] == 1 : label.append('r')
    else : label.append('g')

def tsne(X, n_components,k):
    model_tsne = TSNE(n_components=2,init='pca',random_state=k)
    return model_tsne.fit_transform(X)

def plot_scatter(x, labels, title, txt = False):
    #plt.title(title)
    ax = plt.subplot()
    ax.scatter(x[:,0], x[:,1], c = labels)
    #ax.scatter(x[:,0],s=30,c='red',label='C1')
    #ax.scatter(x[:,1],s=30,c='blue',label='C2')
    
    plt.savefig(title)
    #plt.show()


#for k in range(10):
layer_tsne = tsne(recon, 2,0)
plot_scatter(layer_tsne, label,'fig1_5.jpg')









































































































































































