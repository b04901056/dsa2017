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
import torchvision.transforms as transforms
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter 

parser =  argparse.ArgumentParser(description='vae model')
parser.add_argument('-b',type=int,dest='batch_size',required=True)
parser.add_argument('-e',type=int,dest='epoch',required=True)
parser.add_argument('-tn',type=int,dest='train_num',required=True)
parser.add_argument('-lat',type=int,dest='latent_size',required=True)
parser.add_argument('-lam',type=float,dest='lambda_kl',required=True)
parser.add_argument('-tm',type=int,dest='test_num',required=True)
parser.add_argument('-exp',type=str,dest='exp',required=True)
parser.add_argument('-tsne',type=int,dest='tsne',required=True)
args = parser.parse_args()
writer = SummaryWriter('runs/exp_'+args.exp)

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
'''
data = os.listdir('test')
data.sort()
data = data[0]
im = Image.open('test/'+data)
print('test/'+data)
input()
im = (np.array(im)/127.5)-1
print(im.shape)
input()
print(im) 
input()
#im = transforms.ToTensor()(im)
#im = torch.FloatTensor(im).view(64,3,64)
im = torch.FloatTensor(im).permute(2,0,1)
#im = transforms.ToTensor()(im)
print(im) 
input()
torchvision.utils.save_image(im,'save.jpg')
print('---')
input()
'''

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

 
def mse_loss(inp, target):
    out = (inp - target)**2
    loss = out.sum() # or sum over whatever dimensions
    return loss/(inp.shape[1]*inp.shape[2]*inp.shape[3]) 
#reconstruction_function = nn.MSELoss()

def loss_function(recon_x, x, mu, logvar):
    #MSE = reconstruction_function(recon_x, x)
    MSE = mse_loss(recon_x, x)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5) 
    #KLD = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)
    return MSE + KLD*args.lambda_kl , MSE , KLD 

model = vae(nc=3, ngf=64, ndf=64, latent_size = args.latent_size ).cuda()
#print(model)
#input()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

#training_set = get_data('train',num=args.train_num,name='train',batch_size=args.batch_size,shuffle=True)
#testing_set = get_data('test',num=2621,name='test',batch_size=args.batch_size,shuffle=False)
#print('saved data ...')
#input()

'''
arr = np.load('train.npy')[:args.test_num]
#print(arr)
#input()
print('loaded training set')
print(arr.shape)
arr = torch.FloatTensor(arr)                                                                                                           
dataset = imagedataset(arr,'train')
training_set = DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
'''
'''
arr = np.load('test.npy')
#print('loaded testing set')
#print(arr.shape)
arr = torch.FloatTensor(arr)                                                                                                           
dataset = imagedataset(arr,'test')
testing_set = DataLoader(dataset,batch_size=args.batch_size,shuffle=False)
'''
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

kld_curve = []
mse_curve = []

def train(epoch,iteration):
    model.train()
    train_loss = 0
    start = time.time()
    kld_loss = 0
    mse_loss = 0
    for step , (batch_x,batch_y,_) in enumerate(training_set):
        batch_size = len(batch_x) 
        batch_idx = step + 1  
        batch_x = Variable(batch_x).cuda()
        batch_y = Variable(batch_y).cuda()
        optimizer.zero_grad()
        recon,mu,logvar = model(batch_x)
        loss , MSE , KLD = loss_function(recon,batch_x,mu,logvar)
        loss.backward()
        train_loss+=loss.data[0]
        optimizer.step()
        kld_loss += KLD / batch_size  
        mse_loss += MSE / batch_size 
        #print('batch_idx=',batch_idx)
        #print(epoch*len(training_set))
        #input()   
        #writer.add_scalar('MSE_train', MSE.data[0] / len(batch_x), (epoch-1)*len(training_set)+batch_idx)
        #writer.add_scalar('KLD_train', KLD.data[0] / len(batch_x), (epoch-1)*len(training_set)+batch_idx)
        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.6f} | MSE: {:.6f} | KLD: {:.6f} | Time: {} '.format(
                   epoch 
                   , batch_idx * batch_size 
                   , len(training_set.dataset) 
                   , 100. * batch_idx * batch_size / len(training_set.dataset)
                   , loss.data[0] / batch_size 
                   , MSE.data[0] / batch_size 
                   , KLD.data[0] / batch_size 
                   , timeSince(start, batch_idx * batch_size / len(training_set.dataset)))
                   , end='')

        if batch_idx % 100 == 0 :
            kld_curve.append(kld_loss/100)
            mse_curve.append(mse_loss/100)
            writer.add_scalar('MSE_loss', mse_loss/100 , iteration )
            writer.add_scalar('KLD_loss', kld_loss/100 , iteration )
            kld_loss = 0
            mse_loss = 0
            iteration += 1    
    
    print('\n ====> Epoch: {} | Time: {} | Average loss: {:.4f}'.format(
        epoch 
        , timeSince(start,1)
        , train_loss / len(training_set.dataset)))

    return iteration 
 
testing_mse_loss = []
 
def test(epoch):
    model.eval()
    test_mse_loss = 0
    for step , (batch_x,_) in enumerate(testing_set):
        batch_idx = step + 1
        batch_x = Variable(batch_x,volatile=True).cuda()
        recon,mu,logvar = model(batch_x)
        loss , MSE , KLD  = loss_function(recon,batch_x,mu,logvar)
        test_mse_loss+=MSE.data[0]
        writer.add_scalar('MSE_test', MSE.data[0] / len(batch_x), (epoch-1)*len(testing_set)+batch_idx)
        writer.add_scalar('KLD_test', KLD.data[0] / len(batch_x), (epoch-1)*len(testing_set)+batch_idx)
        print('\rTest on testing set: | [{}/{} ({:.0f}%)]  Loss: {:.6f} MSE: {:.6f} KLD: {:.6f} '.format(
                   batch_idx * len(batch_x)
                   , len(testing_set.dataset) 
                   ,100. * batch_idx * len(batch_x) / len(testing_set.dataset)
                   , loss.data[0] / len(batch_x)
                   , MSE.data[0] / len(batch_x)
                   , KLD.data[0] / len(batch_x))
                   , end='')
        
    testing_mse_loss.append(test_mse_loss/len(testing_set))
        
    #print()
    #print('====> Average loss: {:.4f}'.format(
                  #test_loss / len(testing_set.dataset)))
    #print()
    return test_mse_loss / len(testing_set)
 
def gaussian(ins , mean, stddev):
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise
                            
def rand_faces(num,epoch,model):
    model.eval()    
    #z = torch.randn(num*num, args.latent_size)
    #z = torch.zeros(num,args.latent_size)
    z = np.random.normal(0, 1, (32, args.latent_size))
    z = Variable(torch.FloatTensor(z), volatile=True)
    #z = gaussian(z,0,1)
    z = z.cuda()
    recon = model.decode(z).permute(0,3,1,2)
    #print('recon:',recon)
    #input()
    recon = recon.data 
    final_plot = recon
    #print('recon:',recon)
    #input()
    #img = torchvision.utils.make_grid(recon,nrow=num,normalize=True) 
    #writer.add_image(str(epoch)+'_random_sample.jpg', img , epoch)
    final_plot = torchvision.utils.make_grid(final_plot,nrow=num,normalize=True) 
    #print('final_plot:',final_plot)
    #input()
    return final_plot.permute(1,2,0).cpu().numpy()

def rec_face(num,epoch,model):
    model.eval()         
    seq = np.random.randint(2621, size=10).tolist()
    seq = torch.LongTensor(seq)
    z = torch.index_select(testing_set.dataset.img,0,seq)
    img_orig = torchvision.utils.make_grid(z.permute(0,3,1,2),nrow=num,normalize=True)
    print('img_orig:',img_orig)
    input()
    final_plot = z.permute(0,3,1,2)
    z = Variable(z, volatile=True)
    z = z.cuda()
    recon = model.encode(z.permute(0,3,1,2))[0]
    recon = model.decode(recon).permute(0,3,1,2)
    #print('recon:',recon)
    #input()
    recon = recon.data
    print('final_plot:',final_plot)
    print('recon:',recon)
    input()
    final_plot = torch.cat((final_plot,recon.cpu()),0)
    #print('recon:',recon)
    #input()
    img_recon = torchvision.utils.make_grid(recon,nrow=num,normalize=True) 
    final_plot = torchvision.utils.make_grid(final_plot,nrow=num,normalize=True)
    #writer.add_image(str(epoch)+'_original.jpg', img_orig , epoch)
    #writer.add_image(str(epoch)+'_reconstruct.jpg', img_recon , epoch)
    return final_plot.permute(1,2,0).cpu().numpy() 

np.random.seed(2)
'''
model = torch.load('model/vae/model_300.pt')
final_plot = rec_face(10,1,model)
#ori = ori.permute(1,2,0)
#print('ori:',ori.shape)
#input()
#ori_grid= Image.fromarray(((ori+1)*127.5).astype(np.uint8))
scipy.misc.imsave('fig1_3.jpg', ((final_plot+1)*127.5))
#ori_grid.save('fig1_3.jpg')
'''
model = torch.load('model_300.pt')
final_plot = rand_faces(8,1,model)
#ori = ori.permute(1,2,0)
#print('ori:',ori.shape)
#input()
#ori_grid= Image.fromarray(((ori+1)*127.5).astype(np.uint8))
scipy.misc.imsave('fig1_4.jpg', ((final_plot+1)*127.5))












































































































































































