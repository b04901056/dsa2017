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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
    masks = np.empty((n_masks, 512, 512, 7),dtype='uint8')
    for i, file in enumerate(file_list):    
        if file[5]=='s': continue   
        read_path = os.path.join(filepath, file)         
        print(i)                                                                              
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
    return masks.reshape(n_masks,512,512,7)
'''
train_masks = rgb2label(train_path)
print('train_masks size = ',train_masks.shape)
np.save('train_label.npy',train_masks)

val_masks = rgb2label(validation_path)
print('val_masks size = ',val_masks.shape)
np.save('val_label.npy',val_masks)

train = os.listdir(train_path)
for x in train:
    if(x[5]=='m'): continue 
    print(x) 
    read_path = os.path.join(train_path,x)
    tmp = io.imread(read_path) 
    train_sat.append(tmp)

train_input = np.array(train_sat,dtype='uint8')
print('train_input size = ',train_input.shape)
np.save('train_input.npy',train_input)

val = os.listdir(validation_path)
for x in val:
    if(x[5]=='m'): continue
    print(x)
    read_path = os.path.join(validation_path,x)
    tmp = io.imread(read_path)
    val_sat.append(tmp)

val_input = np.array(val_sat,dtype='uint8')
print('val_input size = ',val_input.shape)
np.save('val_input',val_input)
'''
train_input = np.load('train_input.npy')
print('train_input shape = ',train_input.shape)
train_label = np.load('train_label.npy')
print('train_label shape = ',train_label.shape)
val_input = np.load('val_input.npy')
print('val_input shape = ',val_input.shape)
val_label = np.load('val_label.npy')
print('val_label shape = ',val_label.shape)

####################################################################################################################

class EpochSaver(Callback):
    def __init__(self, model):
        self.model = model 

    def on_epoch_end(self, epoch, logs={}):
        if epoch==1 or epoch==10 or epoch==20:
            name = 'model_epoch_' + str(epoch) + '.h5'
            self.model.save(name) 
print('build model ...')

input_shape = (512,512,3) 

img_input = Input(shape=input_shape)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1' )(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2' )(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool' )(x)
#f1 = x
# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1' )(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2' )(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool' )(x)
#f2 = x


#Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1' )(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2' )(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3' )(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool' )(x)
#f3 = x

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1' )(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2' )(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3' )(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool' )(x)
#f4 = x

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1' )(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2' )(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3' )(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool' )(x)
#f5 =

o = x

o = ( Conv2D( 512 , ( 7 , 7 ) , activation='relu' , padding='same'))(o)
o = Dropout(0.5)(o)
o = ( Conv2D( 512 , ( 1 , 1 ) , activation='relu' , padding='same'))(o)
o = Dropout(0.5)(o)

o = ( Conv2D( 7 ,  ( 1 , 1 ) , kernel_initializer='he_normal' ))(o)
o = Conv2DTranspose( 7,  
                     kernel_size=(64,64) ,  
										 strides=(32,32) ,  
                     padding = 'same',
                     activation = 'softmax', 
                     use_bias=False ,
                     name='upsampling')(o)


model = Model( img_input , o )

weight_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
model.load_weights(weight_path,by_name=True)

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9)
#loss_fn = softmax_sparse_crossentropy_ignoring_last_label
loss_fn = 'categorical_crossentropy'

model.summary()
model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=['accuracy'])
model.fit(train_input,
          train_label, 
          batch_size = 6, 
          epochs= 40, 
          verbose= 1,
          validation_data = (val_input,val_label), 
          callbacks=[EpochSaver(model)])

model.save('model_epoch_40.h5')
