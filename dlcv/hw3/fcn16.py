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
import keras.callbacks
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
    masks = np.empty((n_masks, 512, 512,7 ),dtype='uint8')
    for i, file in enumerate(file_list): 
        print(i)
        read_path = os.path.join(filepath, file)                                                                        
        mask = scipy.misc.imread(read_path)                                                                        
        mask = (mask >= 128).astype(int)                                                                                              
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]                                                                  
        masks[i, mask == 3] = [1,0,0,0,0,0,0]  # (Cyan: 011) Urban land                                                            
        masks[i, mask == 6] = [0,1,0,0,0,0,0]  # (Yellow: 110) Agriculture land                                                          
        masks[i, mask == 5] = [0,0,1,0,0,0,0]  # (Purple: 101) Rangeland                                                                  
        masks[i, mask == 2] = [0,0,0,1,0,0,0]  # (Green: 010) Forest land                                                                
        masks[i, mask == 1] = [0,0,0,0,1,0,0]  # (Blue: 001) Water                                                                        
        masks[i, mask == 7] = [0,0,0,0,0,1,0]  # (White: 111) Barren land                                                                 
        masks[i, mask == 0] = [0,0,0,0,0,0,1]  # (Black: 000) Unknown  
        masks[i, mask == 4] = [0,0,0,0,0,0,1]
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
    
    
    model.add(Convolution2D(4096,kernel_size=(3,3),padding = "same",activation = "relu",name = "fc_6"))
    
    #Replacing fully connnected layers of VGG Net using convolutions
    model.add(Convolution2D(4096,kernel_size=(1,1),padding = "same",activation = "relu",name = "fc7"))
    
    
    # Gives the classifications scores for each of the 21 classes including background
    model.add(Convolution2D(7,kernel_size=(1,1),padding="valid", kernel_initializer='he_normal',name = "score_fr"))
    
    
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
    model.add(Cropping2D(cropping=((0,2),(0,2))))
    return model

def FCN_16(image_size) :
    fcn_16 = FCN_16_helper(512)
    
    #Calculating conv size after the sequential block
    #32 if image size is 512*512
    Conv_size = fcn_16.layers[-1].output_shape[2] 
    
    skip_con = Convolution2D(7,kernel_size=(1,1),padding = "same", kernel_initializer='he_normal', name = "score_pool4")
    
    #Addig skip connection which takes adds the output of Max pooling layer 4 to current layer
    Summed = add(inputs = [skip_con(fcn_16.layers[14].output),fcn_16.layers[-1].output])
    
    
    Up = Deconvolution2D(7,kernel_size=(32,32),strides = (16,16),padding = "same",activation = 'softmax',use_bias=False,name = "upsample_new")
    
    #528 if image size is 512*512
    Deconv_size = (Conv_size-1)*16+32
    
    #16 if image size is 512*512
    extra_margin = (Deconv_size - Conv_size*16)
    
    #Cropping to get the original size of the image
    crop = Cropping2D(cropping = ((0,extra_margin),(0,extra_margin)))
    return Model(fcn_16.input, Up(Summed))
train_masks = rgb2label(train_path)
#print('train_masks size = ',train_masks.shape)
np.save('/datadrive/data/train_label.npy',train_masks)

val_masks = rgb2label(validation_path)
#print('val_masks size = ',val_masks.shape)
np.save('/datadrive/data/val_label.npy',val_masks)

train = [file for file in os.listdir(train_path) if file.endswith('.jpg')]
train.sort()
for x in train:  
    read_path = os.path.join(train_path,x)
    print(read_path)
    tmp = io.imread(read_path)/255 
    train_sat.append(tmp)
train_input = np.array(train_sat)
np.save('/datadrive/data/train_input.npy',train_input)

val = [file for file in os.listdir(validation_path) if file.endswith('.jpg')]
val.sort()
for x in val:
    read_path = os.path.join(validation_path,x)
    print(read_path)
    tmp = io.imread(read_path)/255
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
        if epoch%2==0 or epoch==1:
            name = '/datadrive/model_8s_epoch_' + str(epoch) + '.h5'
            self.model.save(name)  
classes = 7
print('build model ...')

input_img = Input(shape=(512, 512, 3))
x_1 = Conv2D(64, (3, 3), activation='selu', padding='same', name='block1_conv1')(input_img)
x_1 = Conv2D(64, (3, 3), activation='selu', padding='same', name='block1_conv2')(x_1)
x_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x_1)

x_2 = Conv2D(128, (3, 3), activation='selu', padding='same', name='block2_conv1')(x_1)
x_2 = Conv2D(128, (3, 3), activation='selu', padding='same', name='block2_conv2')(x_2)
x_2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x_2)

x_3 = Conv2D(256, (3, 3), activation='selu', padding='same', name='block3_conv1')(x_2)
x_3 = Conv2D(256, (3, 3), activation='selu', padding='same', name='block3_conv2')(x_3)
x_3 = Conv2D(256, (3, 3), activation='selu', padding='same', name='block3_conv3')(x_3)
x_3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x_3)

x_4 = Conv2D(512, (3, 3), activation='selu', padding='same', name='block4_conv1')(x_3)
x_4 = Conv2D(512, (3, 3), activation='selu', padding='same', name='block4_conv2')(x_4)
x_4 = Conv2D(512, (3, 3), activation='selu', padding='same', name='block4_conv3')(x_4)
x_4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x_4)

x_5 = Conv2D(512, (3, 3), activation='selu', padding='same', name='block5_conv1')(x_4)
x_5 = Conv2D(512, (3, 3), activation='selu', padding='same', name='block5_conv2')(x_5)
x_5 = Conv2D(512, (3, 3), activation='selu', padding='same', name='block5_conv3')(x_5)
x_5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x_5)

vgg_16 = Model(input_img, x_5)
vgg_16.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

o = Conv2D(4096, (3, 3), activation='selu', padding='same', name='block6_conv1')(x_5)
o = Dropout(0.5)(o)
o = Conv2D(4096, (1, 1), activation='selu', padding='same', name='block6_conv2')(o)
o = Dropout(0.5)(o)

o = Conv2D(classes, (1, 1), padding='valid', kernel_initializer='he_normal', name='score')(o)
o = Conv2DTranspose(classes, (4, 4), strides=(2, 2), padding='valid', name='block6_upsample')(o)
o = Cropping2D(cropping=((0, 2), (0, 2)))(o)

pool4 = Conv2D(classes, (1, 1), padding='same', kernel_initializer='he_normal', name='score_pool4')(x_4)
merge = Add()([o, pool4])
o = Conv2DTranspose(classes, (32, 32), strides=(16, 16), padding='same', activation='softmax',
                            use_bias=False, name='block7_upsample')(merge)

model = Model(input_img, o)
#weight_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
#model.load_weights(weight_path,by_name=True)

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

