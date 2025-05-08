from tensorflow.keras.layers import *
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
import tensorflow as tf

import numpy as np


def binary_crossentropy_with_retina_loss(gamma):
    def binary_crossentropy_with_retina_loss1(y_true, y_pred):
        # gamma = 2
        prob = K.tf.where(K.greater(y_true,0.5), 1-y_pred, y_pred)+K.epsilon()
        # prob = K.tf.where(K.greater(prob, 0), prob, K.zeros_like(prob))
        prob = prob**gamma
        return K.mean(prob*K.binary_crossentropy(y_pred, y_true), axis=-1)
    return binary_crossentropy_with_retina_loss1


def build_net(currPtchSz, num_channels = 1, flag_batch_norm = True,padding='valid',num_filters = 64,
              which_loss='binary_crossentropy', addWeightsReg = False, gamma = 0):
    """" from Learning to Compare Image Patches via Convolutional Neural Networks, Zagoruyko
            2ch: 2ch = C(96; 7; 3)-ReLU-P(2; 2)-C(192; 5; 1)-ReLU-
            P(2; 2)-C(256; 3; 1)-ReLU-F(256)-ReLU-F(1) """

    # set net name
    net_name = 'pointNet_ptchsz{}_{}'.format(currPtchSz,which_loss)

    if which_loss == 'BCE_retina_loss':
        net_name = net_name + '_gamma{}'.format(gamma)

    # init network
    net = tf.keras.models.Sequential(name=net_name)
    net.add(tf.keras.layers.InputLayer(input_shape=(currPtchSz, currPtchSz, num_channels)))

    # Convolution Layers
    # conv1
    if addWeightsReg==False:
        net.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1),kernel_initializer=tf.keras.initializers.glorot_normal(), activation='relu',padding=padding))
    else:
        net.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1),
                                    kernel_initializer=tf.keras.initializers.glorot_normal(), activation='relu',
                                    padding=padding,
                                    kernel_regularizer=l2(0.1),
                                    bias_regularizer=l2(0.1)))
    if currPtchSz>=19:
        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    if flag_batch_norm:
        net.add(BatchNormalization())

    # conv2
    if currPtchSz>=5:
        if addWeightsReg == False:
            net.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1),kernel_initializer=tf.keras.initializers.glorot_normal(), activation='relu',padding=padding))
        else:
            net.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1),
                                        kernel_initializer=tf.keras.initializers.glorot_normal(), activation='relu',
                                        padding=padding,
                                        kernel_regularizer=l2(0.1),
                                        bias_regularizer=l2(0.1)))
        if currPtchSz >= 11:
            net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        if flag_batch_norm:
            net.add(BatchNormalization())

        # conv3
        if currPtchSz >= 7:
            if addWeightsReg == False:
                net.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1),kernel_initializer=tf.keras.initializers.glorot_normal(), activation='relu',padding=padding))
            else:
                net.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(), activation='relu',
                                            padding=padding,
                                            kernel_regularizer=l2(0.1),
                                            bias_regularizer=l2(0.1)))
            if currPtchSz in [25,23,17,15,13,9]:
                net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
            if currPtchSz == 17:
                net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Fully connected
    net.add(tf.keras.layers.Flatten())
    if addWeightsReg == False:
        net.add(tf.keras.layers.Dense(256, activation='relu'))
    else:
        net.add(tf.keras.layers.Dense(256, activation='relu',
                                   kernel_regularizer = l2(0.1),
                                   bias_regularizer = l2(0.1)))
    # net.add(tf.keras.layers.Dense(256, activation='relu'))


    if which_loss == 'binary_crossentropy':
        net.add(tf.keras.layers.Dense(1))
        net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    elif which_loss == 'HingeLoss':
        net.add(tf.keras.layers.Dense(1, activation='tanh'))
        net.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])
    elif which_loss == 'SquaredHingeLoss':
        net.add(tf.keras.layers.Dense(1, activation='tanh'))
        net.compile(optimizer='adam', loss='squared_hinge', metrics=['accuracy'])
    elif which_loss == 'BCE_retina_loss':
        net.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        net.compile(optimizer='adam', loss=binary_crossentropy_with_retina_loss(gamma), metrics=['accuracy'])

    # net.add(tf.keras.layers.Dense(1, activation='linear'))
    # net.add(tf.keras.layers.Dense(1, activation='tanh'))

    # learning_rate=1e-6
    # sgd = tf.keras.optimizers.SGD(lr=learning_rate, decay=1e-3, momentum=0.9, nesterov=False,clipnorm=1.0)

    # net.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])

    return net
########## LOUISA #################
def build_net_tf(currPtchSz, num_channels = 1, flag_batch_norm = True,padding='valid',num_filters = 64,
              which_loss='binary_crossentropy', addWeightsReg = False, gamma = 0):
    """" from Learning to Compare Image Patches via Convolutional Neural Networks, Zagoruyko
            2ch: 2ch = C(96; 7; 3)-ReLU-P(2; 2)-C(192; 5; 1)-ReLU-
            P(2; 2)-C(256; 3; 1)-ReLU-F(256)-ReLU-F(1) """

    # set net name
    net_name = 'pointNet_ptchsz{}_{}'.format(currPtchSz,which_loss)

    if which_loss == 'BCE_retina_loss':
        net_name = net_name + '_gamma{}'.format(gamma)

    # init network
    net = tf.keras.models.Sequential(name=net_name)
    net.add(tf.keras.layers.InputLayer(input_shape=(currPtchSz, currPtchSz, num_channels)))

    # Convolution Layers
    # conv1
    if addWeightsReg==False:
        net.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1),kernel_initializer=tf.keras.initializers.glorot_normal(), activation='relu',padding=padding))
    else:
        net.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1),
                                    kernel_initializer=tf.keras.initializers.glorot_normal(), activation='relu',
                                    padding=padding,
                                    kernel_regularizer=l2(0.1),
                                    bias_regularizer=l2(0.1)))
    if currPtchSz>=19:
        net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    if flag_batch_norm:
        net.add(BatchNormalization())

    # conv2
    if currPtchSz>=5:
        if addWeightsReg == False:
            net.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1),kernel_initializer=tf.keras.initializers.glorot_normal(), activation='relu',padding=padding))
        else:
            net.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1),
                                        kernel_initializer=tf.keras.initializers.glorot_normal(), activation='relu',
                                        padding=padding,
                                        kernel_regularizer=l2(0.1),
                                        bias_regularizer=l2(0.1)))
        if currPtchSz >= 11:
            net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        if flag_batch_norm:
            net.add(BatchNormalization())

        # conv3
        if currPtchSz >= 7:
            if addWeightsReg == False:
                net.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1),kernel_initializer=tf.keras.initializers.glorot_normal(), activation='relu',padding=padding))
            else:
                net.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1),
                                            kernel_initializer=tf.keras.initializers.glorot_normal(), activation='relu',
                                            padding=padding,
                                            kernel_regularizer=l2(0.1),
                                            bias_regularizer=l2(0.1)))
            if currPtchSz in [25,23,17,15,13,9]:
                net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
            if currPtchSz == 17:
                net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # Fully connected
    net.add(tf.keras.layers.Flatten())
    if addWeightsReg == False:
        net.add(tf.keras.layers.Dense(256, activation='relu'))
    else:
        net.add(tf.keras.layers.Dense(256, activation='relu',
                                   kernel_regularizer = l2(0.1),
                                   bias_regularizer = l2(0.1)))
    # net.add(keras.layers.Dense(256, activation='relu'))


    if which_loss == 'binary_crossentropy':
        net.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    elif which_loss == 'HingeLoss':
        net.add(tf.keras.layers.Dense(1, activation='tanh'))
        net.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])
    elif which_loss == 'SquaredHingeLoss':
        net.add(tf.keras.layers.Dense(1, activation='tanh'))
        net.compile(optimizer='adam', loss='squared_hinge', metrics=['accuracy'])
    elif which_loss == 'BCE_retina_loss':
        net.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        net.compile(optimizer='adam', loss=binary_crossentropy_with_retina_loss(gamma), metrics=['accuracy'])

    # net.add(keras.layers.Dense(1, activation='linear'))
    # net.add(keras.layers.Dense(1, activation='tanh'))

    # learning_rate=1e-6
    # sgd = keras.optimizers.SGD(lr=learning_rate, decay=1e-3, momentum=0.9, nesterov=False,clipnorm=1.0)

    # net.compile(optimizer='adam', loss='hinge', metrics=['accuracy'])

    return net
###################################

def build_net_mobile(currPtchSz, num_channels = 1, flag_batch_norm = True, padding='valid', num_filters = 64):

    # init network
    net = tf.keras.models.Sequential(name='pointNet_ptchsz{}_{}'.format(currPtchSz,'mobile'))
    net.add(tf.keras.layers.InputLayer(input_shape=(currPtchSz, currPtchSz, num_channels)))

    # layer 1
    net.add(tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1),
                                kernel_initializer=tf.keras.initializers.glorot_normal(), activation='relu',
                                padding=padding))
    if flag_batch_norm:
        net.add(BatchNormalization())

    # layer 2 - mobile block
    net.add(tf.keras.layers.SeparableConv2D(num_filters, (3, 3),activation='relu'))

    # layer 3 - mobile block
    net.add(tf.keras.layers.SeparableConv2D(num_filters, (3, 3),activation='relu'))
    net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # layer 4 - mobile block
    net.add(tf.keras.layers.SeparableConv2D(num_filters, (3, 3),activation='relu'))
    net.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    # layer 5
    net.add(tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1),
                                kernel_initializer=tf.keras.initializers.glorot_normal(), activation='relu',
                                padding=padding))

    # FC
    net.add(tf.keras.layers.Flatten())
    net.add(tf.keras.layers.Dense(256, activation='relu'))
    net.add(tf.keras.layers.Dense(1, activation='sigmoid'))



    # compile
    net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return net

# def mobile_block(net,num_depthwise_filt,num,pointwise_filt):
#     net.add(keras.layers.SeparableConv2D(num_depthwise_filt, (3, 3)))



def res_block(x_in, num_filters, flag_batch_norm = True):
    # 2 layer residual block
    x = Conv2D(num_filters, 3, padding='same')(x_in)
    x = Activation('relu')(x)
    if flag_batch_norm==True:
        x = BatchNormalization()(x)
    x = Conv2D(num_filters, 3, padding='same')(x)
    if flag_batch_norm == True:
        x = BatchNormalization()(x)
    x = Concatenate()([x_in, x])

    return x


def res_block_3(x_in, num_filters, flag_batch_norm = True):
    # 3 layer residual block 1x1,3x3,1x1
    x = Conv2D(num_filters, 3, padding='same')(x_in)
    x = Activation('relu')(x)
    if flag_batch_norm==True:
        x = BatchNormalization()(x)
    x = Conv2D(num_filters, 3, padding='same')(x)
    if flag_batch_norm == True:
        x = BatchNormalization()(x)
    x = Concatenate()([x_in, x])

    return x

def build_net_res_net(currPtchSz, num_channels = 1, flag_batch_norm = True, num_filters = 64, which_loss='binary_crossentropy'):

    kernel_size = (3,3)
    input = Input(shape=(currPtchSz, currPtchSz, num_channels))

    # Layer 1 - change from 2 channels to 64
    x = Conv2D(filters = 16,
               kernel_size = kernel_size,
               strides = (1, 1),
               activation ='relu')(input)
    if flag_batch_norm==True:
        x = BatchNormalization()(x)

    # add res blocks
    x = res_block(x, 16, flag_batch_norm = flag_batch_norm)
    # if currPtchSz>=19:
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = res_block(x, 32, flag_batch_norm=flag_batch_norm)
    # if currPtchSz >= 11:
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = res_block(x, 64, flag_batch_norm=flag_batch_norm)
    if currPtchSz>=11:
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
        if currPtchSz>=19:
            x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    net = tf.keras.models.Model(input, x, name='pointNet_resNet_ptchSz{}'.format(currPtchSz))

    net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return net


def check_num_weights(net):

    num_weights = 0
    all_weights = net.get_weights()
    for curr_weights in all_weights:
        curr_weights_num = 1
        for numWeigts in curr_weights.shape:
            curr_weights_num = curr_weights_num*numWeigts
        num_weights = num_weights + curr_weights_num
    return num_weights


if __name__ == "__main__":

    patch_sz_row = 25
    patch_sz_col = 25
    num_channels = 2
    flag_batch_norm = False
    padding = 'valid'
    num_filters = 64

    # ptchSz = np.arange(25,2,-2)
    ptchSz = [7,13,25]
    for currPtchSz in ptchSz:
        print('Working on patch size {}'.format(currPtchSz))
        net = build_net(currPtchSz,
                  num_channels=num_channels,
                  flag_batch_norm=flag_batch_norm,
                  padding=padding,
                  num_filters=num_filters,
                  which_loss = 'binary_crossentropy')
        # net = build_net_res_net(currPtchSz,
        #                         num_channels=num_channels,
        #                         flag_batch_norm=True,
        #                         num_filters=64,
        #                         which_loss='binary_crossentropy')
        # net = build_net_mobile(currPtchSz,
        #                         num_channels=num_channels,
        #                         flag_batch_norm=True,
        #                         num_filters=64)
        # num_weights = check_num_weights(net)
        # print('# weights: {}'.format(num_weights))
        net.summary()
        del net