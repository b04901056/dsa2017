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
    def __init__(self,img,attr,mode):
        self.img = img
        self.attr = attr 
        self.mode = mode 
    def __getitem__(self,i):
        if self.mode == 'train':
            x = self.img[i]
            label = self.attr[i]
            return x,label 
        elif self.mode == 'test':
            x = self.img[i]
            return x
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

        self.e6 = nn.Conv2d(512,64,4,2,1, bias=False) #(64,1,1)
        #self.bn6 = nn.BatchNorm2d(1024)

        self.aux_linear = nn.Linear(64,1)
        self.disc_linear = nn.Linear(64,1)

        self.sigmoid = nn.Sigmoid()
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = x.permute(0,3,1,2)
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h6 = self.leakyrelu(self.e6(h5))
        h6 = h6.view(-1,64) 
        #print('h6:',h6) 
        #input()
        c = self.aux_linear(h6)
        #print('c:',c)
        #input()
        c = self.sigmoid(c)
        #print('c:',c)
        #input()
        s = self.disc_linear(h6)
        s = self.sigmoid(s)
        #print('s:',s)
        #input()

        return s,c

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
print(net_D)
print('-'*50)
print(net_G)
print('-'*50) 
optimizerD = optim.Adam(net_D.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizerG = optim.Adam(net_G.parameters(), lr=0.0001, betas=(0.5, 0.999))

s_criterion = nn.BCELoss()
c_criterion = nn.BCELoss() 

#training_set = get_data('train',num=args.train_num,name='train',batch_size=args.batch_size,shuffle=True)
#testing_set = get_data('test',num=2621,name='test',batch_size=args.batch_size,shuffle=False)
#print('saved data ...')
#input()

train_label = []
with open('train.csv') as f:
    f.readline()
    for i in range(args.test_num):
        attr = f.readline().replace('\n','').split(',')[1:]
        #print(attr)
        #input()
        a = float(attr[8])
        train_label.append(a)
train_label = np.array(train_label)
#print(train_label) 
train_label = torch.FloatTensor(train_label)
#input()
 
arr = np.load('train.npy')[:args.test_num]
#print(arr)
#input()
print('loaded training set')
print(arr.shape)
arr = torch.FloatTensor(arr)                                                                                                           
dataset = imagedataset(arr,train_label,'train')
training_set = DataLoader(dataset,batch_size=args.batch_size,shuffle=True)

'''
arr = np.load('test.npy')
print('loaded testing set')
print(arr.shape)
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

def train_iter(epoch,D,G,iteration):
    dis_loss = 0
    gen_loss = 0
    D.train()
    G.train()
    start = time.time()
    for step , (batch_x,batch_l) in enumerate(training_set):
        batch_idx = step + 1 
        batch_size = len(batch_x)
        img = Variable(batch_x).cuda()
        label = Variable(batch_l).cuda()
        #print('batch_l:',batch_l)
        #input()
       
    
    # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        
    # train with real data
        D.zero_grad()
        
        label_real = torch.ones(batch_size)
        label_real_var = Variable(label_real.cuda())
        
        D_real_result , c_out = D(img)
        D_real_result = D_real_result.squeeze(1)
        c_out = c_out.squeeze(1)
        #print('label_real_var:',label_real_var)
        #print('D_real_result:',D_real_result)
        #input() 
        D_real_loss_img = s_criterion(D_real_result, label_real_var)
        #print('c_out:',c_out)
        #print('label:',label.float())
        #input()
        D_real_loss_label = c_criterion(c_out,label.float())
        
        D_real_loss = D_real_loss_img + D_real_loss_label 
        #D_real_loss.backward(retain_graph=True) 

    # train with fake data
         
        label_fake = torch.zeros(batch_size)
        label_fake_var = Variable(label_fake.cuda())
        #print('label:',label)
        label = np.random.randint( 0 , 2 , batch_size) 
        #print('label:',label)
        #input()
        noise_ = np.random.normal(0, 1, (batch_size, args.latent_size)) 
        #label_onehot = np.zeros((batch_size, 2))
        #label_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size),0] = label[np.arange(batch_size)]
        
        noise = torch.from_numpy(noise_).float()
        #print('noise:',noise)
        #input()
        noise = Variable(noise).cuda()
        fake = G(noise)
        s_output , c_output = D(fake)
        s_output = s_output.squeeze(1)
        c_output = c_output.squeeze(1)
        c_label = Variable(torch.FloatTensor(label)).cuda()
        #print('c_label:',c_label)
        #input()
        #print('s_output size:',s_output.size())
        #print('label_fake_var size:',label_fake_var.size())
        #input()
        D_fake_loss_img   = s_criterion(s_output, label_fake_var)
        D_fake_loss_label = c_criterion(c_output, c_label)
        
        #print('D_fake_loss_img=',D_fake_loss_img)
        #print('D_fake_loss_label=',D_fake_loss_label)
        #input()

        D_fake_loss = D_fake_loss_img + D_fake_loss_label 
        #D_fake_loss.backward(retain_graph=True)
        
    # update parameter 
        D_train_loss = D_real_loss + D_fake_loss 
        D_train_loss.backward(retain_graph=True) 
        optimizerD.step()
        
    # calculate discriminator accuracy 
        correct_real = 0
        correct_fake = 0
        real_result = D_real_result.data.cpu().numpy()
        fake_result = s_output.data.cpu().numpy()
        for i in range(batch_size):
            if real_result[i] >= 0.5 : correct_real += 1
            if fake_result[i] <= 0.5 : correct_fake += 1
            writer.add_scalar('Discriminator_accuracy_real', float(correct_real) / batch_size , iteration)
            writer.add_scalar('Discriminator_accuracy_fake', float(correct_fake) / batch_size , iteration)

    # Update G network: maximize log(D(G(z)))
        
        G.zero_grad()
        s_output , c_output = D(fake)
        s_output = s_output.squeeze(1)
        c_output = c_output.squeeze(1)
        #print('s_output size:',s_output.size())
        #print('label_real_var size:',label_real_var.size())
        #input()
        G_loss_img = s_criterion(s_output,label_real_var)
        G_loss_label = c_criterion(c_output,c_label)
        
        # update parameter
        #print('G_loss_img:',G_loss_img)
        #input()
        #print('G_loss_label:',G_loss_label)
        #input()

        G_train_loss = G_loss_img + G_loss_label
        #print('G_train_loss:',G_train_loss)
        #input()
        G_train_loss.backward()
        optimizerG.step()
                        
    # print training status
        
        dis_loss += D_train_loss.data[0]
        gen_loss += G_train_loss.data[0]
        writer.add_scalar('Real_classification_loss', D_real_loss.data[0]/ batch_size , iteration)
        writer.add_scalar('Fake_classification_loss', D_fake_loss.data[0]/ batch_size , iteration)
        writer.add_scalar('G_train_loss', G_train_loss.data[0] , iteration)
        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)] | D_Loss: {:.6f} | G_Loss: {:.6f} | step: {} | Time: {} '.format(
                   epoch 
                   , batch_idx * batch_size 
                   , len(training_set.dataset) 
                   , 100. * batch_idx * batch_size / len(training_set.dataset)
                   , D_train_loss.data[0]
                   , G_train_loss.data[0]
                   , iteration
                   , timeSince(start, batch_idx*batch_size/ len(training_set.dataset)))
                   , end='')
        iteration += 1
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
    #label = np.random.randint( 0 , 2 , num) 
    #noise_ = np.random.normal(0, 1, (num, args.latent_size)) 
    #label_onehot_first  = np.zeros((num, 2))
    #label_onehot_second = np.zeros((num, 2))
    #label_onehot_first[np.arange(num), 0]  = 1
    #label_onehot_second[np.arange(num), 1] = 1
    
    noise_first = np.random.normal(0, 1, (num, args.latent_size))
    noise_second = np.copy(noise_first)
    #print('noise_first:',noise_first)
    #print('noise_second',noise_second)
    #input()

    noise_first[np.arange(num), 0] = 0
    noise_second[np.arange(num), 0] = 1
    #print('noise_first:',noise_first)
    #input()
    #print('noise_second:',noise_second)
    #input()


    noise_first  = torch.from_numpy(noise_first).float()
    noise_second = torch.from_numpy(noise_second).float()

    #print('noise_first:',noise_first)
    #input()
    #print('noise_second:',noise_second)
    #input()
    noise_first  = Variable(noise_first, volatile=True).cuda()
    noise_second = Variable(noise_second, volatile=True).cuda()
    
    fake_first  = generator(noise_first).permute(0,3,1,2).data
    fake_second = generator(noise_second).permute(0,3,1,2).data

    img_first = torchvision.utils.make_grid(fake_first,nrow=num,normalize=True)
    writer.add_image(str(epoch)+'_random_sample_a.jpg', img_first , epoch)
    
    img_second = torchvision.utils.make_grid(fake_second,nrow=num,normalize=True)
    writer.add_image(str(epoch)+'_random_sample_b .jpg', img_second , epoch)

    '''
    generator.eval()    
    #z = torch.randn(num*num, args.latent_size)
    z = torch.zeros(num,args.latent_size)
    z = Variable(z, volatile=True)
    z = gaussian(z,0,1)
    z = z.cuda() # generator(z) shape (-1,64,64,3)
    recon = generator(z).permute(0,3,1,2)
    recon = recon.data 
    img = torchvision.utils.make_grid(recon,nrow=num,normalize=True)
    writer.add_image(str(epoch)+'_random_sample.jpg', img , epoch)
    '''
step = 1
for epoch in range(1,args.epoch+1):
 
    step = train_iter(epoch,net_D,net_G,(epoch-1)*len(training_set))

    rand_faces(10,epoch,net_G)

    if epoch%5 == 0 :
        torch.save(net_G,'model_acgan_generator_'+str(epoch)+'.pt')
        #torch.save(net_D,'model_acgan_discriminator_'+str(epoch)+'.pt')
         






























































































































































