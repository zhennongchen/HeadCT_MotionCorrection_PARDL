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
 
import HeadCT_MotionCorrection_PARDL.STN.model_components as compo

conv_dict = {2: Conv2D, 3: Conv3D}
max_pooling_dict = {2: MaxPool2D, 3: MaxPool3D}

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


def get_CNN(nb_filters, dimension = 3, crop_STN = True, crop_size = 30, CP_num = 5):
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

        layers += [Dropout(0.25)(layers[-1])]#

        # 3D
        tx_layer = Dense(128, activation = LeakyReLU(alpha=0.1), use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = 'tx_hidden')(layers[-1])
        tx_layer = Dropout(0.2)(tx_layer)
        tx = Dense(CP_num - 1, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'tx')(tx_layer)

        ty_layer = Dense(128, activation = LeakyReLU(alpha=0.1), use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = 'ty_hidden')(layers[-1])
        ty_layer = Dropout(0.2)(ty_layer)
        ty = Dense(CP_num - 1, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'ty')(ty_layer)

        tz_layer = Dense(128, activation = LeakyReLU(alpha=0.1), use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = 'tz_hidden')(layers[-1])
        tz_layer = Dropout(0.2)(tz_layer)
        tz = Dense(CP_num - 1, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'tz')(tz_layer)
        
        rx_layer = Dense(128, activation = LeakyReLU(alpha=0.1), use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = 'rx_hidden')(layers[-1])
        rx_layer = Dropout(0.2)(rx_layer)
        rx = Dense(CP_num - 1, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'rx')(rx_layer)
        
        ry_layer = Dense(128, activation = LeakyReLU(alpha=0.1), use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = 'ry_hidden')(layers[-1])
        ry_layer = Dropout(0.2)(ry_layer)
        ry = Dense(CP_num - 1, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'ry')(ry_layer)

        rz_layer = Dense(128, activation = LeakyReLU(alpha=0.1), use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = 'rz_hidden')(layers[-1])
        rz_layer = Dropout(0.2)(rz_layer)
        rz = Dense(CP_num - 1, use_bias= False, kernel_initializer = "orthogonal", 
                                    kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'rz')(rz_layer)


        ## 2D:
        # tx_layer = Dense(128, activation = LeakyReLU(alpha=0.1), use_bias= False, kernel_initializer = "orthogonal", 
        #                             kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = 'tx_hidden')(layers[-1])
        # tx_layer = Dropout(0.2)(tx_layer)
        # tx = Dense(4, use_bias= False, kernel_initializer = "orthogonal", 
        #                             kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'tx')(tx_layer)
        
        # ty_layer = Dense(128, activation = LeakyReLU(alpha=0.1), use_bias= False, kernel_initializer = "orthogonal", 
        #                             kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = 'ty_hidden')(layers[-1])
        # ty_layer = Dropout(0.2)(ty_layer)
        # ty = Dense(4, use_bias= False, kernel_initializer = "orthogonal", 
        #                             kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'ty')(ty_layer)

        # theta_layer = Dense(128, activation = LeakyReLU(alpha=0.1), use_bias= False, kernel_initializer = "orthogonal", 
        #                             kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4) , name = 'theta_hidden')(layers[-1])
        # theta_layer = Dropout(0.2)(theta_layer)
        # theta = Dense(4, use_bias= False, kernel_initializer = "orthogonal", 
        #                             kernel_regularizer = l2(1e-4), bias_regularizer = l2(1e-4), name = 'theta')(theta_layer)


        # do B-spline for motion parameters
        # tx_ys = compo.bspline_fit()(tx)
        # ty_ys = compo.bspline_fit()(ty)
        # theta_ys = compo.bspline_fit()(theta)

        # # make transformation matrix (negative)
        # matrix = compo.make_transform_matrix()([-ty_ys , -tx_ys ,theta_ys]) # STN has opposite x and y coordinate & opposite rotation compared to our transformation. so switch tx and ty, and no need to add negative to theta
        # matrix_transpose = tf.transpose(matrix, [1,0,2,3])
        
        # # Bilinear interpolation 
        # input_layer_1 = tf.transpose(tf.expand_dims(input_layer, axis = -1), [0,4,1,2,3,5])
        # image = tf.transpose(tf.stack([input_layer_1[:,:,:,:,5,:], input_layer_1[:,:,:,:,10,:]], axis = 0), [1,2,3,4,0,5]) # (None, 25,128,128,2,1)
        # image = tf.transpose(image,[1,4,0,2,3,5])
        # if crop_STN == True:
        #     array = tf.stack([tf.stack([compo.BilinearInterpolation(height=image.shape[3], width=image.shape[4])([img, matrix_transpose[j,...]])[:,(image.shape[3]//2 - crop_size): (image.shape[3]//2 + crop_size), (image.shape[4]//2 - crop_size): (image.shape[4]//2 + crop_size), :]  for img in image[j,...]],axis = 0) for j in range(0,image.shape[0])], axis = 0)
        # else:
        #     array = tf.stack([tf.stack([compo.BilinearInterpolation(height=image.shape[3], width=image.shape[4])([img, matrix_transpose[j,...]]) for img in image[j,...]],axis = 0) for j in range(0,image.shape[0])], axis = 0)
        # array = tf.transpose(array,[2,0,3,4,1,5])
        # final_image = tf.math.reduce_mean(array,axis = 1)
        # print('array shape: ',array.shape)
        # print('final_image shape: ',final_image.shape)
        
        return tx, ty, tz, rx, ry, rz
    
    return f

