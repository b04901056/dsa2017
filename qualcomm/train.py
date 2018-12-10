from nn import Datamanager,DNN
import sys
import numpy as np
import argparse
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       setting option                           '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
EPOCH = 500
BATCH_SIZE = 256 
parser = argparse.ArgumentParser(description='setting module parameter.') 
parser.add_argument('-train', dest='training_set',type=str,required=True)
parser.add_argument('-test', dest='testing_set',type=str,required=True)  
parser.add_argument('-u', dest='unit',type=int,nargs='+',required=True) 
args = parser.parse_args()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading data                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm=Datamanager() 
sys.stdout.flush()
dm.get_data('train',args.training_set,BATCH_SIZE,args,False) 
dm.get_data('test',args.testing_set,BATCH_SIZE,args,False) 
print('finish loading data')
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       training                                 '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dnn = DNN(args).double().cuda()
print(dnn) 
loss_train = []
loss_val = []
loss_test = []
accu_train = []
accu_val = []
accu_test = []
# training and testing
for epoch in range(1,EPOCH + 1):
	loss , accu = dm.train(dnn,dm.dataset['train'],epoch)
	loss_train.append(loss)
	accu_train.append(accu)

	loss , accu = dm.val(dnn,'Val',dm.dataset['val'])
	loss_val.append(loss)
	accu_val.append(accu)

	loss , accu = dm.val(dnn,'Test',dm.dataset['test'])
	loss_test.append(loss)
	accu_test.append(accu)

	print('-'*50)

np.save('loss_train',np.array(loss_train)) 
np.save('loss_val',np.array(loss_val)) 
np.save('loss_test',np.array(loss_test)) 
np.save('accu_train',np.array(accu_train)) 
np.save('accu_val',np.array(accu_val)) 
np.save('accu_test',np.array(accu_test)) 
