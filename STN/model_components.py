import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


class element_1D(Layer):
    def __init__(self,index):
        super(element_1D, self).__init__()
        self.index = index

    def get_config(self):
        config = super().get_config().copy()
        config.update({'index' : self.index, })
        return config
    
    def call(self, inputs):
        output = inputs[...,self.index]; output = tf.reshape(output,[-1,1])
        return output


class element_2D(Layer):
    def __init__(self,index1, index2):
        super(element_2D, self).__init__()
        self.index1 = index1
        self.index2 = index2

    def get_config(self):
        config = super().get_config().copy()
        config.update({'index1' : self.index1,  'index2' : self.index2,})
        return config
        
    
    def call(self, inputs):
        output = inputs[...,self.index1, self.index2]; output = tf.reshape(output,[-1,1,1])
        return output


class build_x_points(Layer): # x axis of the control point and xs in B-spline
    def __init__(self,num, start, end):
        super(build_x_points, self).__init__()
        self.num = num
        self.start = start
        self.end = end

    def get_config(self):
        config = super().get_config().copy()
        config.update({'num' : self.num, 'start': self.start, 'end': self.end,})
        return config
    
    def call(self, inputs):
        values = tf.linspace(self.start, self.end, self.num)
        inputs_tile = tf.tile(inputs,[1, self.num // inputs.shape[-1]])
        a = tf.ones_like(inputs_tile)   # [None, 5]
        
        x_points = [tf.reshape(a[...,i] * values[i],[-1,1]) for i in range(0,a.shape[-1])]
        x_points = tf.reshape(tf.stack(x_points, axis = -1),[-1,self.num])
        return x_points



class bspline_fit(Layer):
    def __init__(self):
        super(bspline_fit, self).__init__()

    def advance_indexing_2d(self, inputs, I):
        shape = tf.shape(inputs)
        batch_size, length = shape[0], shape[1] # input (batch_size, height, width, channel)
        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1))
        b = tf.tile(batch_idx, (1, length)) # shape (batch_size, height, width)
        indices = tf.stack([b, I], 2) # shape (batch_size, height, width, 3)
        
        return tf.gather_nd(inputs, indices) # Important!: shape(batch_size, height, width, 1), value of these indices in the inputs image
    
    def make_I(self,inputs):
        I = tf.zeros_like(inputs)
        I = tf.cast(I, dtype = tf.int32)
        a = tf.constant([0,0,0,1,1,2,2,3,3,3], dtype = tf.int32)
        I = I+a
        return I

    def h_poly_helper_tf(self,tt0, tt1, tt2, tt3):
  
        A = tf.constant([
            [1, 0, -3, 2],
            [0, 1, -2, 1],
            [0, 0, 3, -2],
            [0, 0, -1, 1]], dtype=tt1.dtype)

        first = A[0,0] * tt0  + A[0,1] * tt1 + A[0,2] * tt2 + A[0,3] * tt3; first = first[:,tf.newaxis,:]
        second = A[1,0] * tt0  + A[1,1] * tt1 + A[1,2] * tt2 + A[1,3] * tt3; second = second[:,tf.newaxis,:]
        third = A[2,0] * tt0  + A[2,1] * tt1 + A[2,2] * tt2 + A[2,3] * tt3; third = third[:,tf.newaxis,:]
        fourth = A[3,0] * tt0  + A[3,1] * tt1 + A[3,2] * tt2 + A[3,3] * tt3; fourth = fourth[:,tf.newaxis,:]
   
        output = tf.concat([first, second, third, fourth], axis = 1)
        return output
    
    def h_poly_tf(self,t):

        tt0 = tf.constant(1.)
        tt1 = tt0 * t # same dimension as t
        tt2 = tf.math.multiply(tt1, t) # same dimension as t
        tt3 = tf.math.multiply(tt2, t)
        return self.h_poly_helper_tf(tt0, tt1, tt2 , tt3)

    def get_I_plus_1(self,x):
        x1 = element_1D(1)(x)
        x2 = element_1D(2)(x)
        x3 = element_1D(3)(x)
        x4 = element_1D(4)(x)
        Iplus1 = tf.concat([x1, x1, x1, x1, x1, x1, x2, x2, x2, x2, x2, x2, x3, x3, x3, x3, x3, x3, x4, x4, x4, x4, x4, x4 , x4],axis = -1)
        return Iplus1

    def get_I(self,x):
        x0 = element_1D(0)(x)
        x1 = element_1D(1)(x)
        x2 = element_1D(2)(x)
        x3 = element_1D(3)(x)
        
        I = tf.concat([x0, x0, x0, x0, x0, x0, x1, x1, x1, x1, x1, x1, x2, x2, x2, x2, x2, x2, x3, x3, x3, x3, x3, x3, x3],axis = -1)
        return I


    def interp_func_tf(self,x, y, xs):
        m = (y[...,1:] - y[...,:-1])/(x[...,1:] - x[...,:-1])
        
        first_term = tf.reshape(m[...,0], [-1,1])
        second_term = tf.reshape((m[...,1:] + m[...,:-1])/2, [-1,3])
        third_term = tf.reshape(m[...,-1], [-1,1])
        m = tf.concat([first_term, second_term, third_term], axis = 1)
            
        # II = tf.searchsorted(x[1:], xs)  # each xs belong to which spline segment
        x_1 = self.get_I_plus_1(x)
        x_2 = self.get_I(x)
        dx = x_1 - x_2
        poly_input = (xs - x_2) / (dx)
        hh = self.h_poly_tf(poly_input)

        term1 = tf.math.multiply(hh[:,0,:],self.get_I(y))
        term2 = tf.math.multiply(tf.math.multiply(hh[:,1,:], self.get_I(m)),dx)
        term3 = tf.math.multiply(hh[:,2,:],self.get_I_plus_1(y))
        term4 = tf.math.multiply(tf.math.multiply(hh[:,3,:],self.get_I_plus_1(m)),dx)
        return  term1 + term2 + term3 + term4

        
    def call(self, inputs): 
        # y = inputs
        y_0 = tf.zeros_like(inputs[...,1]); y_0 = tf.reshape(y_0, [-1,1]); y = tf.concat([y_0, inputs],axis = -1) # add 0 to control points
        x = build_x_points(5, 0. ,1.)(y)
        xs = build_x_points(25, 0.04, 1.)(y)
        output = self.interp_func_tf(x,y,xs)
        # print('b-spline output shape: ', output.shape)
          
        return output



class make_transform_matrix(Layer): 
    def __init__(self):
        super(make_transform_matrix, self).__init__()

    def rotation(self, theta):
        first_row = tf.reshape(tf.concat([tf.math.cos(theta)  , -tf.math.sin(theta), tf.zeros_like(theta)], axis = -1),[-1,1,3])
        second_row = tf.reshape(tf.concat([tf.math.sin(theta)  , tf.math.cos(theta), tf.zeros_like(theta)], axis = -1), [-1,1,3])
        third_row = tf.reshape(tf.concat([tf.zeros_like(theta), tf.zeros_like(theta), tf.ones_like(theta)], axis = -1), [-1,1,3])
        return tf.concat([first_row, second_row, third_row], axis = 1)

    def translation(self,tx,ty):
        first_row = tf.reshape(tf.concat([tf.ones_like(tx) , tf.zeros_like(tx), tx], axis = -1),[-1,1,3])
        second_row = tf.reshape(tf.concat([tf.zeros_like(ty) , tf.ones_like(ty), ty], axis = -1), [-1,1,3])
        third_row = tf.reshape(tf.concat([tf.zeros_like(tx), tf.zeros_like(tx), tf.ones_like(tx)], axis = -1), [-1,1,3])
        return tf.concat([first_row, second_row, third_row], axis = 1)

    def m(self,tx,ty,theta):
        multiply = tf.linalg.matmul(self.rotation(theta), self.translation(tx,ty))
        return multiply[:,0:2,:]

    
    def call(self, inputs):
        tx_ys, ty_ys, theta_ys = inputs

        matrix = [self.m(element_1D(i)(tx_ys), element_1D(i)(ty_ys), element_1D(i)(theta_ys))  for i in range(0, tx_ys.shape[-1])]
        matrix = tf.transpose(tf.stack(matrix,axis = 0),[1,0,2,3])
        return matrix

        # matrix = [tf.concat([tf.reshape(tf.concat([tf.math.cos(element_1D(i)(theta_ys)), -tf.math.sin(element_1D(i)(theta_ys)), element_1D(i)(tx_ys)], axis = -1),[-1,1,3]),
        #                     tf.reshape(tf.concat([tf.math.sin(element_1D(i)(theta_ys)), tf.math.cos(element_1D(i)(theta_ys)), element_1D(i)(ty_ys)], axis = -1),[-1,1,3])], axis = 1 ) for i in range(0,theta_ys.shape[-1])]

        # matrix = tf.transpose(tf.stack(matrix,axis = 0),[1,0,2,3])
        # return matrix


class BilinearInterpolation(tf.keras.layers.Layer):
    def __init__(self, height=40, width=40):
        super(BilinearInterpolation, self).__init__()
        self.height = height
        self.width = width

    def compute_output_shape(self, input_shape):
        return [None, self.height, self.width, 1]

    def get_config(self):
        return {
            'height': self.height,
            'width': self.width,
        }
    
    # def build(self, input_shape):
    #     print("Building Bilinear Interpolation Layer with input shape:", input_shape)

    def advance_indexing(self, inputs, x, y):
        '''
        Numpy like advance indexing is not supported in tensorflow, hence, this function is a hack around the same method
        '''      
      
        shape = tf.shape(inputs)
        batch_size, _, _ = shape[0], shape[1], shape[2] # input (batch_size, height, width, channel)
        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, self.height, self.width)) # shape (batch_size, height, width)
        indices = tf.stack([b, y, x], 3) # shape (batch_size, height, width, 3)
        
        return tf.gather_nd(inputs, indices) # Important!: shape(batch_size, height, width, 1), value of these indices in the inputs image

    def call(self, inputs):
        images, theta = inputs
        homogenous_coordinates = self.grid_generator(batch=tf.shape(images)[0])
        output = self.interpolate(images, homogenous_coordinates, theta)
        
        return output

    def grid_generator(self, batch):
        x = tf.linspace(-1, 1, self.width) # normalization
        y = tf.linspace(-1, 1, self.height)
            
        xx, yy = tf.meshgrid(x, y) # xx shape (height,width)
        xx = tf.reshape(xx, (-1,)) # flatten
        yy = tf.reshape(yy, (-1,))
        homogenous_coordinates = tf.stack([xx, yy, tf.ones_like(xx)]) # add the channel, shape (3, height*width)
        homogenous_coordinates = tf.expand_dims(homogenous_coordinates, axis=0) # shape (1, 3, height*width)
        homogenous_coordinates = tf.tile(homogenous_coordinates, [batch, 1, 1]) # shape (batch_size, 3, height*width)
        homogenous_coordinates = tf.cast(homogenous_coordinates, dtype=tf.float32)
        return homogenous_coordinates
    
    def interpolate(self, images, homogenous_coordinates, theta):

        with tf.name_scope("Transformation"):
            transformed = tf.matmul(theta, homogenous_coordinates) # theta shape (2,3), coordinates shape (batch_size,3,height*width), transformed shape (batch_size, 2, height*width)
            transformed = tf.transpose(transformed, perm=[0, 2, 1])
            transformed = tf.reshape(transformed, [-1, self.height, self.width, 2]) # (batch_size, height, width, 2) 2 here represents x and y
            
            x_transformed = transformed[:, :, :, 0]
            y_transformed = transformed[:, :, :, 1]
    
            # previously we have converted the coordinate system into -1 to 1, here we convert it back.
            x = ((x_transformed + 1.) * tf.cast(self.width, dtype=tf.float32)) * 0.5
            y = ((y_transformed + 1.) * tf.cast(self.height, dtype=tf.float32)) * 0.5
    
        with tf.name_scope("VariableCasting"):
            x0 = tf.cast(tf.math.floor(x), dtype=tf.int32)
            x1 = x0 + 1
            y0 = tf.cast(tf.math.floor(y), dtype=tf.int32)
            y1 = y0 + 1
            
            x0 = tf.clip_by_value(x0, 0, self.width-1)
            x1 = tf.clip_by_value(x1, 0, self.width-1)
            y0 = tf.clip_by_value(y0, 0, self.height-1)
            y1 = tf.clip_by_value(y1, 0, self.height-1)
            x = tf.clip_by_value(x, 0, tf.cast(self.width, dtype=tf.float32)-1.0)
            y = tf.clip_by_value(y, 0, tf.cast(self.height, dtype=tf.float32)-1)

        with tf.name_scope("AdvanceIndexing"):
            Ia = self.advance_indexing(images, x0, y0)
            Ib = self.advance_indexing(images, x0, y1)
            Ic = self.advance_indexing(images, x1, y0)
            Id = self.advance_indexing(images, x1, y1)

        with tf.name_scope("Interpolation"): # bilinear interpolation
            x0 = tf.cast(x0, dtype=tf.float32)
            x1 = tf.cast(x1, dtype=tf.float32)
            y0 = tf.cast(y0, dtype=tf.float32)
            y1 = tf.cast(y1, dtype=tf.float32)
                            
            wa = (x1-x) * (y1-y)
            wb = (x1-x) * (y-y0)
            wc = (x-x0) * (y1-y)
            wd = (x-x0) * (y-y0)

            wa = tf.expand_dims(wa, axis=3)
            wb = tf.expand_dims(wb, axis=3)
            wc = tf.expand_dims(wc, axis=3)
            wd = tf.expand_dims(wd, axis=3)
                        
        return tf.math.add_n([wa*Ia + wb*Ib + wc*Ic + wd*Id])


