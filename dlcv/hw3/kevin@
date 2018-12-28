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
from my_classes import DataGenerator
import matplotlib.pyplot as plt

params = {'dim': (512,512),
          'batch_size': 6,
          'n_classes': 7,
          'n_channels': 3,
          'shuffle': True}

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
        if filepath=='train' : labels_train['train_'+file[:4]] = masks
        elif filepath=='validation' : labels_validation['validation_'+file[:4]] = masks

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
    tmp = io.imread(read_path)/255
    np.save('data/train_'+x[:4]+'.npy',tmp)

val = [file for file in os.listdir('validation') if file.endswith('.jpg')]
val.sort()
for x in val:
    read_path = os.path.join('validation',x)
    part_validation.append('validation_'+x[:4])
    print(read_path)
    tmp = io.imread(read_path)/255
    np.save('data/validation_'+x[:4]+'.npy',tmp)

partition['train'] = part_train
partition['validation'] = part_validation

training_generator = DataGenerator(partition['train'], labels_train, **params)
validation_generator = DataGenerator(partition['validation'], labels_validation, **params) 

class EpochSaver(Callback):
    def __init__(self, model):
        self.model = model 

    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0 or epoch == 1 :
            name = 'model_epoch_' + str(epoch) + '.h5'
            self.model.save(name) 

            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig('accu_'+str(epoch)+'.png')
            plt.show()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig('loss_'+str(epoch)+'.png')
            plt.show()

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
x = BatchNormalization()(x)
#f2 = x


#Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1' )(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2' )(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3' )(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool' )(x)
x = Dropout(0.25)(x)
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
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
#f5 =

o = x

o = ( Conv2D( 4096 , ( 3 , 3 ) , activation='relu' , padding='same'))(o)
o = Dropout(0.5)(o)
o = ( Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same'))(o)
o = Dropout(0.5)(o)
o = BatchNormalization()(o)

o = ( Conv2D( 7 ,  ( 1 , 1 ) , padding = 'valid' , kernel_initializer='he_normal' ))(o)
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

#optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9)
optimizer = 'adam'
#loss_fn = softmax_sparse_crossentropy_ignoring_last_label
loss_fn = 'categorical_crossentropy'

model.summary()
model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=['accuracy']) 

history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    steps_per_epoch = len(training_generator),
                    validation_steps = len(validation_generator),
                    use_multiprocessing=True,
                    workers=6, 
                    epochs=100, 
                    callbacks=[EpochSaver(model)])
 
model.save('model_epoch_100.h5')

