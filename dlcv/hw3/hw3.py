import numpy as np
from skimage import io 
import scipy
import os
import sys 
from keras.models import Model
from keras.regularizers import l2
from keras.layers import *
from keras.optimizers import SGD, Adam, Nadam
from keras.engine import Layer
from keras.applications.vgg16 import *
from keras.models import *
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K
from keras.callbacks import Callback
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train_path = 'train'
validation_path = 'validation'
 
train_input = []
val_input = []

train_sat = []
val_sat = []


def rgb2label(filepath):
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    masks = np.empty((n_masks, 512, 512 ),dtype='uint8')
    for i, file in enumerate(file_list): 
        print(i)
        read_path = os.path.join(filepath, file)                                                                        
        mask = scipy.misc.imread(read_path)                                                                        
        mask = (mask >= 128).astype(int)                                                                                              
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]                                                                  
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land                                                            
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land                                                          
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland                                                                  
        masks[i, mask == 2] = 3  # (Green: 010) Forest land                                                                
        masks[i, mask == 1] = 4  # (Blue: 001) Water                                                                        
        masks[i, mask == 7] = 5  # (White: 111) Barren land                                                                 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown  
        masks[i, mask == 4] = 6
    return masks

def Convblock(channel_dimension, block_no, no_of_convs) :
    Layers = []
    for i in range(no_of_convs) :
        
        Conv_name = "block_"+str(block_no)+"_conv"+str(i+1)
        
        # A constant kernel size of 3*3 is used for all convolutions
        Layers.append(Convolution2D(channel_dimension,kernel_size = (3,3),padding = "same",activation = "relu",name = Conv_name))
    
    Max_pooling_name = "block"+str(block_no)+"_pool"
    
    #Addding max pooling layer
    Layers.append(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name = Max_pooling_name))
    
    return Layers

def FCN_16_helper(image_size):          
    model = Sequential()
    model.add(Permute((1,2,3),input_shape = (image_size,image_size,3)))
    
    
    for l in Convblock(64,1,2) :
        model.add(l)
    
    for l in Convblock(128,2,2):
        model.add(l)
    
    for l in Convblock(256,3,3):
        model.add(l)
    
    for l in Convblock(512,4,3):
        model.add(l)
    
    for l in Convblock(512,5,3):
        model.add(l)
    
    
    model.add(Convolution2D(4096,kernel_size=(7,7),padding = "same",activation = "relu",name = "fc_6"))
    
    #Replacing fully connnected layers of VGG Net using convolutions
    model.add(Convolution2D(4096,kernel_size=(1,1),padding = "same",activation = "relu",name = "fc7"))
    
    
    # Gives the classifications scores for each of the 21 classes including background
    model.add(Convolution2D(7,kernel_size=(1,1),padding="same",activation="relu",name = "score_fr"))
    
    
    Conv_size = model.layers[-1].output_shape[2] #16 if image size if 512
    print(Conv_size)
    
    model.add(Deconvolution2D(7,kernel_size=(4,4),strides = (2,2),padding = "valid",activation=None,name = "score2"))
    
    # O = ((I-K+2*P)/Stride)+1 
    # O = Output dimesnion after convolution
    # I = Input dimnesion
    # K = kernel Size
    # P = Padding
    
    # I = (O-1)*Stride + K 
    Deconv_size = model.layers[-1].output_shape[2] #34 if image size is 512*512
    
    print(Deconv_size)
    # 2 if image size is 512*512
    Extra = (Deconv_size - 2*Conv_size)
    
    print(Extra)
    
    #Cropping to get correct size
    model.add(Cropping2D(cropping=((0,Extra),(0,Extra))))
    return model

def FCN_16(image_size) :
    fcn_16 = FCN_16_helper(512)
    
    #Calculating conv size after the sequential block
    #32 if image size is 512*512
    Conv_size = fcn_16.layers[-1].output_shape[2] 
    
    skip_con = Convolution2D(7,kernel_size=(1,1),padding = "same",activation=None, name = "score_pool4")
    
    #Addig skip connection which takes adds the output of Max pooling layer 4 to current layer
    Summed = add(inputs = [skip_con(fcn_16.layers[14].output),fcn_16.layers[-1].output])
    
    
    Up = Deconvolution2D(7,kernel_size=(32,32),strides = (16,16),padding = "valid",activation = None,name = "upsample_new")
    
    #528 if image size is 512*512
    Deconv_size = (Conv_size-1)*16+32
    
    #16 if image size is 512*512
    extra_margin = (Deconv_size - Conv_size*16)
    
    #Cropping to get the original size of the image
    crop = Cropping2D(cropping = ((0,extra_margin),(0,extra_margin)))
    return Model(fcn_16.input, crop(Up(Summed)))
train_masks = rgb2label(train_path)
#print('train_masks size = ',train_masks.shape)
np.save('/datadrive/data/train_label.npy',train_masks)

val_masks = rgb2label(validation_path)
#print('val_masks size = ',val_masks.shape)
np.save('/datadrive/data/val_label.npy',val_masks)

train = [file for file in os.listdir(train_path) if file.endswith('.jpg')]
for x in train:  
    read_path = os.path.join(train_path,x)
    print(read_path)
    tmp = io.imread(read_path) 
    train_sat.append(tmp)
train_input = np.array(train_sat)
np.save('/datadrive/data/train_input.npy',train_input)

val = [file for file in os.listdir(validation_path) if file.endswith('.jpg')]
for x in val:
    read_path = os.path.join(validation_path,x)
    print(read_path)
    tmp = io.imread(read_path)
    val_sat.append(tmp)
val_input = np.array(val_sat)
np.save('/datadrive/data/val_input',val_input)

train_input = np.load('/datadrive/data/train_input.npy')
print('train_input shape = ',train_input.shape)
train_label = np.load('/datadrive/data/train_label.npy')
print('train_label shape = ',train_label.shape)
val_input = np.load('/datadrive/data/val_input.npy')
print('val_input shape = ',val_input.shape)
val_label = np.load('/datadrive/data/val_label.npy')
print('val_label shape = ',val_label.shape)

####################################################################################################################

class EpochSaver(Callback):
    def __init__(self, model):
        self.model = model 

    def on_epoch_end(self, epoch, logs={}):
        if epoch%2==0: and epoch==1:
            name = '/datadrive/model_8s_epoch_' + str(epoch) + '.h5'
            self.model.save(name)  

print('build model ...')

model = FCN_16(512)

weight_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
model.load_weights(weight_path,by_name=True)

optimizer = Adam(lr=0.0001)
loss_fn = 'categorical_crossentropy'

model.summary()
model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=['accuracy']) 

log_filepath = '/tmp/keras_log'
tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath)

model.fit(train_input,
          train_label, 
          batch_size = 6, 
          epochs= 100, 
          verbose= 1,
          validation_data = (val_input,val_label), 
          callbacks=[EpochSaver(model),tb_cb])
 
model.save('model_epoch_100.h5')
