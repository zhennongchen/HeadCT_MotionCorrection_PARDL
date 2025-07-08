# tutorial: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import random
import HeadCT_motion_correction_PAR.Data_processing as Data_processing
import HeadCT_motion_correction_PAR.Defaults as Defaults
import tensorflow as tf
import HeadCT_motion_correction_PAR.functions_collection as ff
import math
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):

    def __init__(self,X_par_image,Y_motion_param,
        model_type = None, # either 'CNN' or 'LSTM'
        patient_num = None, 
        batch_size = None, 
        input_dimension = None,
        output_dimension = None,
        shuffle = None,
        add_noise = False,
        noise_sigma = None,
        seed = 10):

        self.X_par_image = X_par_image
        self.Y_motion_param = Y_motion_param
        self.model_type = model_type
        self.patient_num = patient_num
        self.batch_size = batch_size
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.shuffle = shuffle
        self.add_noise = add_noise
        self.noise_sigma = noise_sigma
        self.seed = seed

        self.on_epoch_end()
        
    def __len__(self):
        
        return self.X_par_image.shape[0]// self.batch_size

    def on_epoch_end(self):
        
        self.seed += 1

        patient_list = np.random.permutation(self.patient_num)
                
        self.indices = np.asarray(patient_list)
        # print('all indexes: ', self.indices,len(self.indices))

    def __getitem__(self,index):
        'Generate one batch of data'
        total_cases = self.patient_num 
        
        current_index = (index * self.batch_size) % total_cases
        if total_cases > current_index + self.batch_size:   # the total number of cases is adequate for next loop
            current_batch_size = self.batch_size
        else:
            current_batch_size = total_cases - current_index  # approaching to the tend, not adequate, should reduce the batch size
       
        indexes = self.indices[current_index : current_index + current_batch_size]
        
        # print('indexes in this batch: ',indexes)

        # allocate memory
        batch_x = np.zeros(tuple([current_batch_size]) + self.input_dimension)
        batch_y1 = np.zeros(tuple([current_batch_size]) + self.output_dimension)
        batch_y2 = np.zeros(tuple([current_batch_size]) + self.output_dimension)
        batch_y3 = np.zeros(tuple([current_batch_size]) + self.output_dimension)

        for i,j in enumerate(indexes):
            # path to input
            x = self.X_par_image[j]
            # adapt image
            if self.model_type[0:2] == 'LS': # LSTM
                x = Data_processing.adapt(x,cutoff = False, add_noise = self.add_noise, sigma = self.noise_sigma, normalize = True, expand_dim= True)
            elif self.model_type[0:2] == 'CN': # CNN
                x = Data_processing.adapt(x,cutoff = False, add_noise = self.add_noise, sigma = self.noise_sigma, normalize = True, expand_dim= False)
                x = np.transpose(x, [1,2,3,0])

            # path to output
            y = self.Y_motion_param[j]
            # print('input y path: ',y)
            y = np.load(y,allow_pickle = True)
            tx = y[0,:][0][1:5] # only need the last four control points, first one is always 0
            tx = ff.convert_translation_control_points(tx, self.input_dimension[1], from_pixel_to_1 = True) # convert to a space -1 ~ 1
            tx = np.asarray(tx) / 2
            ty = y[2,:][0][1:5]
            ty = ff.convert_translation_control_points(ty, self.input_dimension[1], from_pixel_to_1 = True)
            ty = np.asarray(ty) / 2
            theta = y[4,:][0][1:5]
            # print('theta: ',theta)

            batch_x[i] = x
            batch_y1[i] = tx
            batch_y2[i] = ty
            batch_y3[i] = theta

        return batch_x, [batch_y1, batch_y2, batch_y3]