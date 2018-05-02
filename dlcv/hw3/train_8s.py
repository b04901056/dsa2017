from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Activation, Reshape, Dense, Dropout, Flatten, UpSampling2D, BatchNormalization, Add
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, History, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
import numpy as np
import pickle as pk
import sys

lable_num = 7
# abs_path = './drive/colab/DLCV2018SPRING/hw3/'
abs_path = ''

img_input = Input(shape=(512, 512, 3))
trainable_or_not = True
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=trainable_or_not)(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=trainable_or_not)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=trainable_or_not)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=trainable_or_not)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=trainable_or_not)(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=trainable_or_not)(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=trainable_or_not)(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

s8 = Conv2D(lable_num, (1, 1), activation='relu', padding='same', name='8sforward')(x)

y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=trainable_or_not)(x)
y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=trainable_or_not)(y)
y = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=trainable_or_not)(y)
y = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(y)

s16 = Conv2D(lable_num, (1, 1), activation='relu', padding='same', name='16sforward')(y)

z = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=trainable_or_not)(y)
z = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=trainable_or_not)(z)
z = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=trainable_or_not)(z)
z = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(z)

vgg_model = Model(img_input, z)
weights_path = abs_path + 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
vgg_model.load_weights(weights_path, by_name=True)
# vgg_model.trainable = False
z = Conv2D(4096, (3,3), activation='relu', padding='same', name='fc1')(z)
x = Dropout(0.3)(x)
z = Conv2D(4096, (1,1), activation='relu', padding='same', name='fc2')(z)
x = Dropout(0.3)(x)
z = Conv2D(lable_num, (1,1), activation='linear', padding='valid', kernel_initializer='he_normal')(z)

z = Conv2DTranspose(lable_num, kernel_size=4, strides=2, use_bias=False, activation='linear', padding='same')(z)
z = Add()([z, s16])
z = Conv2DTranspose(lable_num, kernel_size=4, strides=2, use_bias=False, activation='linear', padding='same')(z)
z = Add()([z, s8])
z = Conv2DTranspose(lable_num, kernel_size=16, strides=8, use_bias=False, activation='softmax', padding='same')(z)
# z = Activation('softmax')(z)
model = Model(img_input, z)

# model = load_model('/mnt/8s_model-21-0.4248.h5')
model.summary()
#optimizer = SGD(momentum=0.0, lr=1e-4)
optimizer = Adam(lr=0.0001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

val_sat = np.load(abs_path+'npy/val_sat_uint8.npy')
val_label = np.load(abs_path+'npy/val_masks_uint8.npy')
print(val_label.shape)
print('val data loaded!')
train_sat = np.load(abs_path+'npy/train_sat_uint8.npy')
train_label = np.load(abs_path+'npy/train_masks_uint8.npy')
print(train_label.shape)
print('train data loaded!')

###### only for testing
# train_sat = np.load('npy/mini_train_sat.npy')[:2]
# train_label = np.load('npy/mini_train_masks.npy')[:2]
# print('train data loaded!')
######

mode = '8s4'
print('training with mode '+mode)
checkpointer = ModelCheckpoint(filepath='/mnt/'+mode+'_model-{epoch:02d}-{val_loss:.4f}-{val_acc:.3f}.h5', verbose=0, save_best_only=True, period=1)
earlystopping = EarlyStopping(monitor='val_acc', patience=10, min_delta=0.00)
model.fit(train_sat, train_label, batch_size=12, epochs=50, verbose=1, 
          validation_data=(val_sat, val_label),
          callbacks=[checkpointer, earlystopping])
