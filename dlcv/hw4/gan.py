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
from tensorboardX import SummaryWriter 

parser =  argparse.ArgumentParser(description='vae model')
parser.add_argument('-b',type=int,dest='batch_size',required=True)
parser.add_argument('-e',type=int,dest='epoch',required=True)
parser.add_argument('-tn',type=int,dest='train_num',required=True)
parser.add_argument('-lat',type=int,dest='latent_size',required=True)
parser.add_argument('-lam',type=float,dest='lambda_kl',required=True)
parser.add_argument('-tm',type=int,dest='test_num',required=True)
parser.add_argument('-exp',type=str,dest='exp',required=True)
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

        self.e6 = nn.Conv2d(512,512,4,2,1) #(1024,1,1)
        self.bn6 = nn.BatchNorm2d(512)
 
        self.mu = nn.Linear(512,latent_size)
        self.sigma = nn.Linear(512,latent_size) # 512 

        self.d1 = nn.Linear(latent_size, 1024) # (1024,1,1)
        
        self.up1 = nn.ConvTranspose2d(1024,512,4,2,1) 
        self.bn6 = nn.BatchNorm2d(512, 1.e-3)

        self.up2 = nn.ConvTranspose2d(512,256,4,2,1)
        self.bn7 = nn.BatchNorm2d(256, 1.e-3)

        self.up3 = nn.ConvTranspose2d(256,128,4,2,1)
        self.bn8 = nn.BatchNorm2d(128, 1.e-3)

        self.up4 = nn.ConvTranspose2d(128,64,4,2,1)
        self.bn9 = nn.BatchNorm2d(64, 1.e-3)
 
        self.up5 = nn.ConvTranspose2d(64,64,4,2,1)
        self.bn10 = nn.BatchNorm2d(64, 1.e-3)
 
        self.up6 = nn.ConvTranspose2d(64,3,4,2,1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def encode(self,x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        #print(h5.size())
        #input()
        h6 = self.leakyrelu(self.bn6(self.e6(h5)))
 
        h6 = h6.view(-1,512)
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
        h2 = self.leakyrelu(self.bn6(self.up1(h1)))
        #print('h2:',h2.size())
        h3 = self.leakyrelu(self.bn7(self.up2(h2)))
        #print('h3:',h3.size())
        h4 = self.leakyrelu(self.bn8(self.up3(h3)))
        #print('h4:',h4.size())
        h5 = self.leakyrelu(self.bn9(self.up4(h4)))
        #print('h5:',h5.size())
        h6 = self.leakyrelu(self.bn10(self.up5(h5)))
        #print('h6:',h6.size())
        h7 = self.sigmoid(self.up6(h6))
        #print('h7:',h7.size())
        return h7.view(-1,64,64,3)
        
    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self,x):
        mu, logvar = self.encode(x.view(-1, 3, 64 , 64))
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

reconstruction_function = nn.MSELoss(size_average=False)
def loss_function(recon_x, x, mu, logvar):
    MSE = reconstruction_function(recon_x, x)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    #KLD = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)
    return MSE + KLD*args.lambda_kl , MSE , KLD 

model = vae(nc=3, ngf=64, ndf=64, latent_size = args.latent_size ).cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

#training_set = get_data('train',num=args.train_num,name='train',batch_size=args.batch_size,shuffle=True)
#testing_set = get_data('test',num=2621,name='test',batch_size=args.batch_size,shuffle=False)
#print('saved data ...')
#input()

arr = np.load('train.npy')[:args.test_num]
#print(arr)
#input()
print('loaded training set')
print(arr.shape)
arr = torch.FloatTensor(arr)                                                                                                           
dataset = imagedataset(arr,'train')
training_set = DataLoader(dataset,batch_size=args.batch_size,shuffle=True)

arr = np.load('test.npy')
print('loaded testing set')
print(arr.shape)
arr = torch.FloatTensor(arr)                                                                                                           
dataset = imagedataset(arr,'test')
testing_set = DataLoader(dataset,batch_size=args.batch_size,shuffle=False)

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

def train(epoch):
    model.train()
    train_loss = 0
    start = time.time()
    for step , (batch_x,batch_y,_) in enumerate(training_set):
        batch_idx = step + 1 
        batch_x = Variable(batch_x).cuda()
        batch_y = Variable(batch_y).cuda()
        optimizer.zero_grad()
        recon,mu,logvar = model(batch_x)
        loss , MSE , KLD = loss_function(recon,batch_x,mu,logvar)
        loss.backward()
        train_loss+=loss.data[0]
        optimizer.step()
        #print('batch_idx=',batch_idx)
        #print(epoch*len(training_set))
        #input()   
        writer.add_scalar('MSE_train', MSE.data[0] / len(batch_x), (epoch-1)*len(training_set)+batch_idx)
        writer.add_scalar('KLD_train', KLD.data[0] / len(batch_x), (epoch-1)*len(training_set)+batch_idx)
        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)] | Loss: {:.6f} | MSE: {:.6f} | KLD: {:.6f} | step: {} | Time: {} '.format(
                   epoch 
                   , batch_idx * len(batch_x)
                   , len(training_set.dataset) 
                   , 100. * batch_idx * len(batch_x) / len(training_set.dataset)
                   , loss.data[0] / len(batch_x) 
                   , MSE.data[0] / len(batch_x) 
                   , KLD.data[0] / len(batch_x)
                   , (epoch-1)*len(training_set)+batch_idx 
                   , timeSince(start, batch_idx*len(batch_x)/ len(training_set.dataset)))
                   , end='')
    print('\n ====> Epoch: {} | Time: {} | Average loss: {:.4f}'.format(
        epoch 
        , timeSince(start,1)
        , train_loss / len(training_set.dataset)))
    return train_loss / len(training_set)
 
def test(epoch):
    model.eval()
    test_loss = 0
    for step , (batch_x,_) in enumerate(testing_set):
        batch_idx = step + 1
        batch_x = Variable(batch_x,volatile=True).cuda()
        recon,mu,logvar = model(batch_x)
        loss , MSE , KLD  = loss_function(recon,batch_x,mu,logvar)
        test_loss+=loss.data[0]
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
    print()
    print('====> Average loss: {:.4f}'.format(
                  test_loss / len(testing_set.dataset)))
    print()
    return test_loss / len(testing_set)

def gaussian(ins , mean, stddev):
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise
                            
def rand_faces(model,num,epoch):
    model.eval()    
    #z = torch.randn(num*num, args.latent_size)
    z = torch.zeros(num,args.latent_size)
    z = Variable(z, volatile=True)
    #print('z=',z)
    #input()
    z = gaussian(z,0,1)
    #print('z=',z)
    #input()
    z = z.cuda()
    recon = model.decode(z).view(-1,3,64,64)
    #print('recon:',recon)
    #input()
    recon = (recon.data+1)*127.5 
    #print('recon:',recon)
    #input()
    img = torchvision.utils.make_grid(recon,nrow=num)
    writer.add_image(str(epoch)+'_random_sample.jpg', img , epoch)

def rec_face(model,num,epoch):
    model.eval()
    #seq = [i for i in range(2621)]
    seq = np.random.randint(2621, size=num).tolist()
    seq = torch.LongTensor(seq)
    z = torch.index_select(testing_set.dataset.img,0,seq)
    #print('z:',z.size())
    #input()
    z = Variable(z, volatile=True)
    z = z.cuda()
    recon = model(z)[0].view(-1,3,64,64)
    recon = (recon.data+1)*127.5
    img = torchvision.utils.make_grid(recon,nrow=num)
    writer.add_image(str(epoch)+'_reconstruct.jpg', img , epoch)


for epoch in range(1,args.epoch+1):
    train_loss = train(epoch)
    test_loss = test(epoch)
    rand_faces(model,10,epoch)
    rec_face(model,10,epoch)
    if epoch%50 == 0 :
        torch.save(model,'model_'+str(epoch)+'.pt')
         






























































































































































