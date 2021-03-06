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
        im = Image.open(os.path.join(path,x))
        im = np.array(im)
        #im = (im/127.5)-1
        arr.append(im)
    arr = np.array(arr)
    print('saving data ...')
    np.save(name+'.npy',arr)
    arr = torch.FloatTensor(arr)
    dataset = imagedataset(arr,name)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)

class discriminator(nn.Module):
    def __init__(self,nc,ngf,ndf,latent_size):
        super(discriminator,self).__init__()
        self.nc = nc #3
        self.ngf = ngf #64
        self.ndf = ndf #64
        self.latent_size = latent_size 
        #(3,64,64)
        self.e1 = nn.Conv2d(3,32,4,2,1, bias=False) #(32,32,32)
        self.bn1 = nn.BatchNorm2d(32)

        self.e2 = nn.Conv2d(32,64,4,2,1, bias=False) #(64,16,16)
        self.bn2 = nn.BatchNorm2d(64)
 
        self.e3 = nn.Conv2d(64,128,4,2,1, bias=False) #(128,8,8)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.e4 = nn.Conv2d(128,256,4,2,1, bias=False) #(256,4,4)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.e5 = nn.Conv2d(256,512,4,2,1, bias=False) #(512,2,2)
        self.bn5 = nn.BatchNorm2d(512)

        self.e6 = nn.Conv2d(512,1,4,2,1, bias=False) #(1024,1,1)
        #self.bn6 = nn.BatchNorm2d(1024)

        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.2)
    
    def forward(self,x):
        x = x.permute(0,3,1,2)
        #print('x size: ',x.size())
        #input()
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h6 = self.sigmoid(self.e6(h5))
 
        h6 = h6.view(-1,1).squeeze(1)
        return h6 

class generator(nn.Module):
    def __init__(self,nc,ngf,ndf,latent_size):
        super(generator,self).__init__()
        self.nc = nc #3
        self.ngf = ngf #64
        self.ndf = ndf #64
        self.latent_size = latent_size 

        self.d1 = nn.Linear(latent_size, 1024) # (1024,1,1)
        
        self.up1 = nn.ConvTranspose2d(1024,512,4,2,1, bias=False) # (512,2,2)
        self.bn6 = nn.BatchNorm2d(512, 1.e-3)

        self.up2 = nn.ConvTranspose2d(512,256,4,2,1, bias=False) # (256,4,4)
        self.bn7 = nn.BatchNorm2d(256, 1.e-3)

        self.up3 = nn.ConvTranspose2d(256,128,4,2,1, bias=False) # (128,8,8)
        self.bn8 = nn.BatchNorm2d(128, 1.e-3)

        self.up4 = nn.ConvTranspose2d(128,64,4,2,1, bias=False) # (64,16,16)
        self.bn9 = nn.BatchNorm2d(64, 1.e-3)
 
        self.up5 = nn.ConvTranspose2d(64,32,4,2,1, bias=False) # (32,32,32)
        self.bn10 = nn.BatchNorm2d(32, 1.e-3)
 
        self.up6 = nn.ConvTranspose2d(32,3,4,2,1, bias=False) # (3,64,64)
        self.bn11 = nn.BatchNorm2d(3)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.tanh = nn.Tanh()

    def forward(self,x):
        h0 = self.leakyrelu(self.d1(x)) # (1024)
        h0 = h0.view(-1,1024,1,1)
        h1 = self.leakyrelu(self.bn6(self.up1(h0)))
        h2 = self.leakyrelu(self.bn7(self.up2(h1)))
        h3 = self.leakyrelu(self.bn8(self.up3(h2)))
        h4 = self.leakyrelu(self.bn9(self.up4(h3)))
        h5 = self.leakyrelu(self.bn10(self.up5(h4)))
        h6 = self.tanh(self.bn11(self.up6(h5)))

        return h6.permute(0,2,3,1) 
 
net_D = discriminator(nc=3, ngf=64, ndf=64, latent_size = args.latent_size ).cuda()
net_G = generator(nc=3, ngf=64, ndf=64, latent_size = args.latent_size ).cuda() 
#print(net_D)
#print('-'*50)
#print(net_G)
#print('-'*50) 
optimizerD = optim.Adam(net_D.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizerG = optim.Adam(net_G.parameters(), lr=0.0001, betas=(0.5, 0.999))

criterion = nn.BCELoss()

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

dis_accu_real_curve = []
dis_accu_fake_curve = []
d_real_loss_curve = []
d_fake_loss_curve = []

def train_iter(epoch,D,G,iteration):
    dis_loss = 0
    gen_loss = 0
    D.train()
    G.train()
    d_real_loss = 0
    d_fake_loss = 0
    dis_accu_real = 0
    dis_accu_fake = 0
    start = time.time()
    for step , (batch_x,batch_y,_) in enumerate(training_set):
        batch_idx = step + 1 
        batch_x = Variable(batch_x).cuda()
        batch_size = len(batch_x) 
    # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            
        # train with real data
        label_real = torch.ones(batch_size)
        label_real_var = Variable(label_real.cuda())
        D_real_result = D(batch_x)
        D_real_loss = criterion(D_real_result, label_real_var)

        # train with fake data
        label_fake = torch.zeros(batch_size)
        label_fake_var = Variable(label_fake.cuda())
 
        noise = torch.randn(batch_size, args.latent_size)#.view(-1, args.latent_size, 1, 1)
        noise_var = Variable(noise.cuda())
        G_result = G(noise_var)
        D_fake_result = D(G_result)
        D_fake_loss = criterion(D_fake_result, label_fake_var)

        # calculate discriminator accuracy 
        correct_real = 0
        correct_fake = 0
        real_result = D_real_result.data.cpu().numpy()
        fake_result = D_fake_result.data.cpu().numpy()
        for i in range(batch_size):
            #print(real_result[i])
            #print(fake_result[i])
            #input()
            if real_result[i] >= 0.5 : correct_real += 1
            if fake_result[i] <= 0.5 : correct_fake += 1
        #writer.add_scalar('Discriminator_accuracy_real', float(correct_real/len(batch_x)) , iteration)
        #writer.add_scalar('Discriminator_accuracy_fake', float(correct_fake/len(batch_x)) , iteration)

        # update parameter 
        D_train_loss = D_real_loss + D_fake_loss
        optimizerD.zero_grad()
        D_train_loss.backward()
        optimizerD.step()

        d_real_loss += D_real_loss.data[0]
        d_fake_loss += D_fake_loss.data[0]
        dis_accu_real += float(correct_real / batch_size )
        dis_accu_fake += float(correct_fake / batch_size ) 
    # Update G network: maximize log(D(G(z)))
        
        noise = torch.randn(( batch_size , args.latent_size))#.view(-1, args.latent_size, 1, 1)
        noise_var = Variable(noise.cuda())
        G_result = G(noise_var)
        D_fake_result = D(G_result)
        
        # update parameter
        G_train_loss = criterion(D_fake_result, label_real_var)
        optimizerG.zero_grad() 
        G_train_loss.backward()
        optimizerG.step()


    # print training status
        
        dis_loss += D_train_loss.data[0]
        gen_loss += G_train_loss.data[0]
        #writer.add_scalar('D_loss_real', D_real_loss.data[0] , iteration)
        #writer.add_scalar('D_loss_fake', D_fake_loss.data[0] , iteration)
        #writer.add_scalar('G_loss', G_train_loss.data[0] , iteration)
        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)] | D_Loss: {:.6f} | G_Loss: {:.6f} | step: {} | Time: {} '.format(
                   epoch 
                   , batch_idx * batch_size 
                   , len(training_set.dataset) 
                   , 100. * batch_idx * batch_size / len(training_set.dataset)
                   , D_train_loss.data[0]
                   , G_train_loss.data[0]
                   , iteration
                   , timeSince(start, batch_idx* batch_size / len(training_set.dataset)))
                   , end='')
        iteration += 1
        
        if batch_idx % 100 == 0:
            d_real_loss_curve.append(d_real_loss/100)
            d_fake_loss_curve.append(d_fake_loss/100)
            dis_accu_real_curve.append(dis_accu_real/100) 
            dis_accu_fake_curve.append(dis_accu_fake/100)
            writer.add_scalar('d_real_loss_curve',d_real_loss/100 , iteration)    
            writer.add_scalar('d_fake_loss_curve',d_fake_loss/100 , iteration)    
            writer.add_scalar('dis_accu_real',dis_accu_real/100 , iteration)    
            writer.add_scalar('dis_accu_fake',dis_accu_fake/100 , iteration)    
            d_real_loss = 0
            d_fake_loss = 0
            dis_accu_real = 0
            dis_accu_fake = 0
    print('\n ====> Epoch : {} | Time: {} | D_loss: {:.4f} | G_loss: {:.4f} \n'.format(
                epoch 
                , timeSince(start,1)
                , dis_loss / len(training_set)
                , gen_loss / len(training_set) ))
    return iteration

def gaussian(ins , mean, stddev):
    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    return ins + noise
                                    
def rand_faces(num,epoch,generator):
    generator.eval()    
    #z = torch.randn(num*num, args.latent_size)
    #z = torch.zeros(num,args.latent_size)
    #z = Variable(z, volatile=True)
    #z = gaussian(z,0,1)
    z = np.random.normal(0, 1, (32, args.latent_size))
    z = Variable(torch.FloatTensor(z), volatile=True)
    z = z.cuda() # generator(z) shape (-1,64,64,3)
    recon = generator(z).permute(0,3,1,2)
    recon = recon.data 
    img = torchvision.utils.make_grid(recon,nrow=num,normalize=True)
    #writer.add_image(str(epoch)+'_random_sample.jpg', img , epoch)
    recon = torchvision.utils.make_grid(recon,nrow=num,normalize=True)
    return recon.permute(1,2,0).cpu().numpy()

'''
for epoch in range(1,args.epoch+1):
    np.random.seed(1)
    step = train_iter(epoch,net_D,net_G,(epoch-1)*len(training_set)) 
    
    #net_D = torch.load('model/dcgan/model_discriminator_140.pt')
    #net_G = torch.load('model/dcgan/model_generator_140.pt')
    
    rand_faces(10,epoch,net_G)

    if epoch%5 == 0 and epoch > 30 :
        torch.save(net_G,'model_generator_'+str(epoch)+'.pt')
        np.save('d_real_loss_curve',np.array(d_real_loss_curve))    
        np.save('d_fake_loss_curve',np.array(d_fake_loss_curve))    
        np.save('dis_accu_real_curve',np.array(dis_accu_real_curve))    
        np.save('dis_accu_fake_curve',np.array(dis_accu_fake_curve))    
'''
np.random.seed(2)
model = torch.load('model_generator_90.pt')                                                                                                                             
final_plot = rand_faces(8,1,model)
scipy.misc.imsave('fig2_3.jpg', ((final_plot+1)*127.5))






























































































































































