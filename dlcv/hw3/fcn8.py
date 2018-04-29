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
from my_classes import DataGenerator
import matplotlib.pyplot as plt

params = {'dim': (512,512),
          'batch_size': 6,
          'n_classes': 7,
          'n_channels': 3,
          'shuffle': True}
IMAGE_ORDERING = 'channels_first' 
nClasses = 7
partition = {}
labels_train = {}
labels_validation = {}

part_train = []
part_validation = []

def rgb2label(filepath):
    file_list = [file for file in os.listdir(filepath) if file.endswith('.png')]
    file_list.sort()
    n_masks = len(file_list)
    #masks = np.empty((512, 512),dtype=int)
    for file in file_list: 
        read_path = os.path.join(filepath, file)       
        print(read_path)                                                                 
        mask = scipy.misc.imread(read_path)                                                                         
        mask = (mask >= 128).astype(int)                                                                                              
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2] 
        masks = np.empty((512, 512),dtype='uint8')                                                                 
        masks[mask == 3] = 0  # (Cyan: 011) Urban land                                                            
        masks[mask == 6] = 1  # (Yellow: 110) Agriculture land                                                          
        masks[mask == 5] = 2  # (Purple: 101) Rangeland                                                                  
        masks[mask == 2] = 3  # (Green: 010) Forest land                                                                
        masks[mask == 1] = 4  # (Blue: 001) Water                                                                        
        masks[mask == 7] = 5  # (White: 111) Barren land                                                                 
        masks[mask == 0] = 6  # (Black: 000) Unknown  
        masks[mask == 4] = 6
        #print('validation_'+file[:4])
        #print(masks)
        #input()
        if filepath=='train' : labels_train['train_'+file[:4]] = masks
        elif filepath=='validation' : labels_validation['validation_'+file[:4]] = masks
def crop( o1 , o2 , i  ):
    o_shape2 = Model( i  , o2 ).output_shape
    outputHeight2 = o_shape2[2]
    outputWidth2 = o_shape2[3]

    o_shape1 = Model( i  , o1 ).output_shape
    outputHeight1 = o_shape1[2]
    outputWidth1 = o_shape1[3]

    cx = abs( outputWidth1 - outputWidth2 )
    cy = abs( outputHeight2 - outputHeight1 )

    if outputWidth1 > outputWidth2:
        o1 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o1)
    else:
        o2 = Cropping2D( cropping=((0,0) ,  (  0 , cx )), data_format=IMAGE_ORDERING  )(o2)
	
    if outputHeight1 > outputHeight2 :
        o1 = Cropping2D( cropping=((0,cy) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o1)
    else:
		    o2 = Cropping2D( cropping=((0, cy ) ,  (  0 , 0 )), data_format=IMAGE_ORDERING  )(o2)

    return o1 , o2 

rgb2label('train') 
rgb2label('validation') 

#print(labels_train)
#input()
train = [file for file in os.listdir('train') if file.endswith('.jpg')]
train.sort()
for x in train:  
    read_path = os.path.join('train',x)
    part_train.append('train_'+x[:4])
    print(read_path)
    tmp = io.imread(read_path).reshape(3,512,512)/255
    np.save('data_fcn8/train_'+x[:4]+'.npy',tmp)

val = [file for file in os.listdir('validation') if file.endswith('.jpg')]
val.sort()
for x in val:
    read_path = os.path.join('validation',x)
    part_validation.append('validation_'+x[:4])
    print(read_path)
    tmp = io.imread(read_path).reshape(3,512,512)/255
    np.save('data_fcn8/validation_'+x[:4]+'.npy',tmp)

partition['train'] = part_train
partition['validation'] = part_validation

training_generator = DataGenerator(partition['train'], labels_train, **params)
validation_generator = DataGenerator(partition['validation'], labels_validation, **params) 

class EpochSaver(Callback):
    def __init__(self, model):
        self.model = model 

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0 or epoch == 1 :
            name = 'model_epoch_i_fcn8_' + str(epoch) + '.h5'
            self.model.save(name) 

print('build model ...')

input_shape = (3,512,512)
img_input = Input(shape=input_shape)

x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
f1 = x
# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
f2 = x

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
f3 = x

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
f4 = x

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
f5 = x

o = f5

o = ( Conv2D( 4096 , ( 3 , 3 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
o = Dropout(0.5)(o)
o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same', data_format=IMAGE_ORDERING))(o)
o = Dropout(0.5)(o)

o = ( Conv2D( nClasses ,  ( 1 , 1 ) , padding = 'valid' , kernel_initializer='he_normal', data_format=IMAGE_ORDERING ))(o)
o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , data_format=IMAGE_ORDERING )(o)

o2 = f4
o2 = ( Conv2D( nClasses ,  ( 1 , 1 ) , padding = 'valid' , kernel_initializer='he_normal', data_format=IMAGE_ORDERING))(o2)
	
o , o2 = crop( o , o2 , img_input )
	
o = Add()([ o , o2 ])
o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, data_format=IMAGE_ORDERING )(o)
o2 = f3 
o2 = ( Conv2D( nClasses ,  ( 1 , 1 ), padding = 'valid' ,kernel_initializer='he_normal' , data_format=IMAGE_ORDERING))(o2)
o2 , o = crop( o2 , o , img_input )
o  = Add()([ o2 , o ])


o = Conv2DTranspose( nClasses,  
                     kernel_size=(16,16) ,  
                     strides=(8,8) ,  
                     padding = 'same',
                     activation = 'softmax', 
                     use_bias=False ,
                     data_format=IMAGE_ORDERING, 
                     name='upsampling')(o)


model = Model( img_input , o )

weight_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
model.load_weights(weight_path,by_name=True)

#optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9)
optimizer = 'adam'
#loss_fn = softmax_sparse_crossentropy_ignoring_last_label
loss_fn = 'categorical_crossentropy'

model.summary()
model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=['accuracy']) 

log_filepath = '/tmp/keras_log'
tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath)

history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    steps_per_epoch = len(training_generator),
                    validation_steps = len(validation_generator),
                    use_multiprocessing=True,
                    workers=6, 
                    epochs=100, 
                    callbacks=[EpochSaver(model),tb_cb])
 
model.save('model_epoch_100.h5')

