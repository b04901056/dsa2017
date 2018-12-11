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
parser.add_argument('-train', dest='training_set',type=str,required=True)	## Training set file
parser.add_argument('-test', dest='testing_set',type=str,required=True)  	## Testing set file
parser.add_argument('-u', dest='unit',type=int,nargs='+',required=True) 	## Set DNN layer size
args = parser.parse_args()
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       reading data                             '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dm=Datamanager() 															## Create Datamanager object
sys.stdout.flush()
dm.get_data('train',args.training_set,BATCH_SIZE,args,True) 				## Read in training data
dm.get_data('test',args.testing_set,BATCH_SIZE,args,False) 					## Read in testing data
print('finish loading data')
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'''''''''''''''''''''       training                                 '''
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
dnn = DNN(args).double().cuda()												## Set up DNN and put it on GPU using cuda()
print(dnn) 																	## Print DNN model structure
 
for epoch in range(1,EPOCH + 1):											## Training and testing
	loss = dm.train(dnn,dm.dataset['train'],epoch)
	loss = dm.val(dnn,'Val',dm.dataset['val']) 
	loss = dm.val(dnn,'Test',dm.dataset['test']) 

	print('-'*50) 