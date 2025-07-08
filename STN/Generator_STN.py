# tutorial: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import numpy as np
import random
import HeadCT_MotionCorrection_PARDL.Data_processing as dp
import HeadCT_MotionCorrection_PARDL.Defaults as Defaults
import HeadCT_MotionCorrection_PARDL.STN.Bspline as Bspline
import HeadCT_MotionCorrection_PARDL.motion_simulator.transformation as transform
import HeadCT_MotionCorrection_PARDL.functions_collection as ff
import tensorflow as tf
import math
import nibabel as nb
import os
from tensorflow.keras.utils import Sequence



class DataGenerator(Sequence):

    def __init__(self,X_par_image,Y_motion_param, 
        start_slice_list,
        end_slice_list,
        start_slice_sampling = None,
        # Y_true_image,  # for 2D
        patient_num = None, 
        batch_size = None, 
        input_dimension = None,
        output_vector_dimension = None,
        output_img_dimension = None,  # for 2D, if crop size = 30, this is [60,60,2]
        shuffle = None,
        augment = None,
        augment_frequency = None,
        seed = 10):

        self.X_par_image = X_par_image
        self.Y_motion_param = Y_motion_param
        # self.Y_true_image = Y_true_image
        
        self.start_slice_list = start_slice_list
        self.end_slice_list = end_slice_list
        self.start_slice_sampling = start_slice_sampling

        self.patient_num = patient_num
        self.batch_size = batch_size
        self.input_dimension = input_dimension
        self.output_vector_dimension = output_vector_dimension
        self.output_img_dimension = output_img_dimension
        self.shuffle = shuffle
        self.augment = augment
        self.augment_frequency = augment_frequency
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
        batch_y1 = np.zeros(tuple([current_batch_size]) + self.output_vector_dimension)
        batch_y2 = np.zeros(tuple([current_batch_size]) + self.output_vector_dimension)
        batch_y3 = np.zeros(tuple([current_batch_size]) + self.output_vector_dimension)
        batch_y4 = np.zeros(tuple([current_batch_size]) + self.output_vector_dimension)
        batch_y5 = np.zeros(tuple([current_batch_size]) + self.output_vector_dimension)
        batch_y6 = np.zeros(tuple([current_batch_size]) + self.output_vector_dimension)
        # batch_y4 = np.zeros(tuple([current_batch_size]) + self.output_img_dimension + (1,))
    

        for i,j in enumerate(indexes):
            # generate augmentations:
            if self.augment == True:
                do_or_not = np.random.rand()
                if do_or_not > (1-self.augment_frequency):
                    do = 1
                    # augment_t_or_r = np.random.rand()
                    # augment_t_or_r >0.5: # augment translation #
                    augment_tx = np.random.rand() * 8; augment_tx = [augment_tx if np.random.rand() > 0.5 else -augment_tx for i in range(0,1)][0]
                    augment_ty = np.random.rand() * 8; augment_ty = [augment_ty if np.random.rand() > 0.5 else -augment_ty for i in range(0,1)][0]
                    augment_r = 0
                    print('augment tx and ty: ', augment_tx, augment_ty)
                    # else: # augment rotation
                        # augment_tx = 0; augment_ty = 0
                        # augment_r = np.random.rand() * 10; augment_r = [augment_r if np.random.rand() > 0.5 else -augment_r for i in range(0,1)][0] / 180 * np.pi
                else:
                    do = 0

            # path to input
            # x = self.X_par_image[j]
            # x = nb.load(x).get_fdata().astype(np.float)

            # pick slices
            if self.start_slice_sampling is None:
                start_slice = self.start_slice_list[j]
            else:
                start_slice = np.random.choice(self.start_slice_sampling)

            print('start_slice: ',start_slice)
            
            if start_slice == 60:
                start_slice = 55
            if start_slice == 120:
                start_slice = 115
            if start_slice == 180:
                start_slice = 175
            
            # need to change the code if using PARs_ds_crop.nii.gz or PARs_ds_crop_anneal.nii.gz
            x_file = os.path.join(os.path.dirname(self.X_par_image[j]) , 'PARs_slice_' + str(start_slice)+'.nii.gz') 
            x = nb.load(x_file).get_fdata().astype(np.float)
        
            # cut off the background intensity
            x = dp.cutoff_intensity(x, -7000)

            
            if self.augment == True and do == 1:
                par_aug = np.copy(x)
                for kk in range(0,par_aug.shape[0]):
                    I = par_aug[kk,...]
                    _,_,_,transformation_matrix = transform.generate_transform_matrix([augment_tx,augment_ty,0],[0,0,augment_r],[1,1,1],I.shape)
                    transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, I.shape)
                    img_new = transform.apply_affine_transform(I, transformation_matrix, 3, cval = np.min(x))
                    par_aug[kk,...] = img_new
                x = np.copy(par_aug)

            x = x / 1000 # normalize
            x = np.transpose(x, [1,2,3,0])
            
            # path to output vectors
            y = self.Y_motion_param[j]
            y = np.load(y,allow_pickle = True)

            # 3D: pixel resolution: x = 1, y = 1, z = 2.5mm
            tx = y[0,:][0][1:  1+ self.output_vector_dimension[0]] / 1 / 5  # only need the last four control points, first one is always 0
            ty = y[1,:][0][1:  1+ self.output_vector_dimension[0]] / 1 / 5 
            tz = y[2,:][0][1: 1 + self.output_vector_dimension[0]] / 2.5 / 2
            rx = np.asarray(y[3,:][0][1: 1+self.output_vector_dimension[0]]) / 5
            ry = np.asarray(y[4,:][0][1: 1+self.output_vector_dimension[0]]) / 5
            rz = np.asarray(y[5,:][0][1: 1+self.output_vector_dimension[0]]) / 5


            # 2D:
            # tx = np.asarray(y[0,:][0][1:5]) / 5  
            # ty = np.asarray(y[2,:][0][1:5]) / 5
            # theta = np.asarray(y[5,:][0][1:5]) / 5

      
            # # path to output image
            # y_img = self.Y_true_image[j]
            # y_img = nb.load(y_img).get_fdata()
            # if self.augment == True and do == 1:
            #     _,_,_,transformation_matrix = transform.generate_transform_matrix([augment_tx,augment_ty,0],[0,0,augment_r],[1,1,1],y_img.shape)
            #     transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, y_img.shape)
            #     y_img = transform.apply_affine_transform(y_img, transformation_matrix,3, cval = np.min(y_img))

            # y_img = y_img[(y_img.shape[0]//2 - 30):(y_img.shape[0]//2 + 30), (y_img.shape[1]//2 - 30):(y_img.shape[1]//2 + 30),[5,10]]
            # y_img = y_img / 1000
            # y_img = np.expand_dims(y_img, axis = -1)
        

            batch_x[i] = x
            batch_y1[i] = tx
            batch_y2[i] = ty
            batch_y3[i] = tz
            batch_y4[i] = rx
            batch_y5[i] = ry
            batch_y6[i] = rz

        return batch_x, [batch_y1, batch_y2, batch_y3, batch_y4, batch_y5, batch_y6]