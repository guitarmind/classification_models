"""

Squeeze and Excitation Networks in Keras
https://github.com/titu1994/keras-squeeze-excite-network

"""
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras import backend as K


def squeeze_excite_block(input, ratio=16):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


"""

Spatial and Channel Squeeze & Excitation Block (scSE)
https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66568

"""
from keras.layers.core import Lambda, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.merge import Add, Multiply

def cse_block(prevlayer):
    mean = Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)
    lin1 = Dense(K.int_shape(prevlayer)[3]//2, activation='relu', kernel_initializer='he_normal', use_bias=False)(mean)
    lin2 = Dense(K.int_shape(prevlayer)[3], activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(lin1)
    x = Multiply()([prevlayer, lin2])
    return x

def sse_block(prevlayer):
    conv = Conv2D(K.int_shape(prevlayer)[3],
                  (1, 1),
                  padding="same",
                  kernel_initializer="he_normal",
                  activation='sigmoid',
                  strides=(1, 1),
                  use_bias=False)(prevlayer)
    conv = Multiply()([prevlayer, conv])
    return conv

def csse_block(x):
    '''
    Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
    https://arxiv.org/abs/1803.02579
    '''
    cse = cse_block(x)
    sse = sse_block(x)
    x = Add()([cse, sse])
    
    return x
