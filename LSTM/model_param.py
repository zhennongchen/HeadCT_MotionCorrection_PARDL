from termios import VLNEXT
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Conv3D, MaxPool2D, MaxPool3D,UpSampling2D,UpSampling3D,
    Reshape,ZeroPadding2D,ZeroPadding3D,
)
from tensorflow.keras.layers import Concatenate,Multiply, Add, LSTM, Dense, Flatten, TimeDistributed, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
import tensorflow.keras.backend as kb

import HeadCT_motion_correction_PAR.Defaults as Defaults

conv_dict = {2: Conv2D, 3: Conv3D}
max_pooling_dict = {2: MaxPool2D, 3: MaxPool3D}
up_sampling_dict = {2: UpSampling2D, 3: UpSampling3D}
zero_sampling_dict = {2: ZeroPadding2D, 3: ZeroPadding3D}

cg = Defaults.Parameters()

def conv_bn_relu_1x(nb_filter, kernel_size, subsample = (1,), dimension = 3, batchnorm = False, activation = False):
    stride = subsample * dimension # stride
    Conv = conv_dict[dimension]

    def f(input_layer):
        x = Conv(
            filters=nb_filter,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="orthogonal",
            kernel_regularizer=l2(1e-4),
            bias_regularizer=l2(1e-4),
            )(input_layer)
        if batchnorm == True:
            x = BatchNormalization()(x)
        if activation == True:
            # x = PReLU()(x)
            x = LeakyReLU(alpha=0.1)(x)
        return x
    return f



def get_LSTM(input_shape, dimension = 3, activate = True, batch_norm = True ):
    kernel_size = (3,) * dimension
    pool_size = (2, 2 , 1)
    MaxPooling = max_pooling_dict[dimension]
    Conv = conv_dict[dimension]


    def f(input_layer):
        
        layers = []
        for i in np.arange(5):
            if i == 0:
                nb_filters = 16
            elif i == 1:
                nb_filters = 32
            else:
                nb_filters = 64

            if i == 0:
                # first layer
                layers += [TimeDistributed(Conv(filters = 8, kernel_size = kernel_size, strides = (1,1,1), padding = "same", use_bias = False,
                            kernel_initializer = "orthogonal", kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4)), 
                            input_shape = input_shape)(input_layer)]

                if batch_norm == True:
                    layers += [TimeDistributed(BatchNormalization())(layers[-1])]
                if activate == True:
                    layers += [TimeDistributed(LeakyReLU(alpha=0.1))(layers[-1])]

            else:
                layers += [TimeDistributed(MaxPooling(pool_size = pool_size))(layers[-1])]

           
            layers += [TimeDistributed(Conv(filters = nb_filters, kernel_size = kernel_size, strides = (1,1,1), padding = "same", use_bias = False,
                            kernel_initializer = "orthogonal", kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4)), 
                            input_shape = input_shape)(layers[-1])]
            if batch_norm == True:
                layers += [TimeDistributed(BatchNormalization())(layers[-1])]  
            if activate == True:
                    layers += [TimeDistributed(LeakyReLU(alpha=0.1))(layers[-1])]             


            layers += [TimeDistributed(Conv(filters = nb_filters, kernel_size = kernel_size, strides = (1,1,1), padding = "same", use_bias = False,
                            kernel_initializer = "orthogonal", kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4)), 
                            input_shape = input_shape)(layers[-1])]
            if batch_norm == True:
                layers += [TimeDistributed(BatchNormalization())(layers[-1])]  
            if activate == True:
                    layers += [TimeDistributed(LeakyReLU(alpha=0.1))(layers[-1])]              

            
        layers += [TimeDistributed(MaxPooling(pool_size = pool_size))(layers[-1])]

        layers += [TimeDistributed(Flatten(name = 'flatten'))(layers[-1])]

        layers += [TimeDistributed(Dense(2048, activation = LeakyReLU(alpha=0.1) ,use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4)), name = "before_LSTM")(layers[-1])]

        layers += [LSTM(2048, activation = 'tanh', recurrent_activation = 'sigmoid', return_sequences = False, 
                        input_shape = (input_shape[0],2048), 
                        dropout = 0.25, name = 'LSTM')(layers[-1])]


        layers += [Dense(256, activation = LeakyReLU(alpha=0.1), use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = 'after_LSTM')(layers[-1])]
        
        layers += [Dropout(0.25)(layers[-1])]

        control_points_tx = Dense(4, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'tx')(layers[-1])
        control_points_ty = Dense(4, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'ty')(layers[-1])
        control_points_theta = Dense(4, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'theta')(layers[-1])
 
        return control_points_tx, control_points_ty, control_points_theta
    
    return f


def get_CNN(nb_filters, dimension = 3):
    kernel_size = (3,) * dimension
    pool_size = (2, 2 , 1)
    MaxPooling = max_pooling_dict[dimension]

    def f(input_layer): # input layer size (None, 128,128,15,25)
        layers = []
        layers += [conv_bn_relu_1x(int(nb_filters[0] / 2), kernel_size, subsample = (1,), dimension = 3, batchnorm = True, activation = True)(input_layer)]
        layers += [conv_bn_relu_1x(nb_filters[0], kernel_size, subsample = (1,), dimension = 3, batchnorm = True, activation = True)(layers[-1])]

        for i in range(0,5):
            layers += [conv_bn_relu_1x(nb_filters[i], kernel_size, subsample = (1,), dimension = 3, batchnorm = True, activation = True)(layers[-1])]
            layers += [conv_bn_relu_1x(nb_filters[i], kernel_size, subsample = (1,), dimension = 3, batchnorm = True, activation = True)(layers[-1])]
            layers += [MaxPooling(pool_size = pool_size)(layers[-1])]

        layers += [Flatten()(layers[-1])]

        layers += [Dropout(0.25)(layers[-1])]


        layers += [Dense(64, activation = LeakyReLU(alpha=0.1), use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = 'hidden')(layers[-1])]
        
        layers += [Dropout(0.25)(layers[-1])]

        control_points_tx = Dense(4, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'tx')(layers[-1])
        control_points_ty = Dense(4, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'ty')(layers[-1])
        control_points_theta = Dense(4, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'theta')(layers[-1])
 
        return control_points_tx, control_points_ty, control_points_theta
    
    return f