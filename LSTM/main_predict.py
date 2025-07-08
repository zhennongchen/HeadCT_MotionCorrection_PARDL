#!/usr/bin/env python

import HeadCT_motion_correction_PAR.Defaults as Defaults
import model_param
import Generator
import HeadCT_motion_correction_PAR.Data_processing as dp
import HeadCT_motion_correction_PAR.functions_collection as ff
from HeadCT_motion_correction_PAR.Build_lists import Build_list
import HeadCT_motion_correction_PAR.motion_simulator.transformation as transform

import os
import numpy as np
import nibabel as nb
import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2

cg = Defaults.Parameters()

model_type = 'CNN'
trial_name = 'CNN_1'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_PAR_downsampled_MORE_Data_dim128_slice0-15.xlsx')
save_folder = os.path.join(cg.predict_dir,trial_name)
ff.make_folder([os.path.dirname(save_folder),save_folder, os.path.join(save_folder,'parameters'), os.path.join(save_folder,'images')])

# build lists
print('Build List...')
b = Build_list.Build(data_sheet)
batch_list = [0]
batch_list, patient_id_list, patient_subid_list, random_name_list, start_slice_list, end_slice_list, motion_free_file_list, motion_free_ds_file_list, motion_file_list, motion_ds_file_list, y_motion_param_predict, x_par_image_predict, _ = b.__build__(batch_list = batch_list)

n = np.arange(0,200,4)
x_par_image_predict = x_par_image_predict[n]
y_motion_param_predict = y_motion_param_predict[n]

# define model list:
model_list = [['3','003']]#, ['1','218']] #[batch, best_epcoh]

Results = []
for m in model_list:    
    print('model is: ',m)
    model_file = ff.find_all_target_files(['*'+m[1]+'.hdf5'],os.path.join(cg.model_dir,trial_name,'models','batch_'+str(m[0])))
    print(model_file[0])
    model_file = model_file[0]

    # create model and load weights
    print('Create Model...') 
    if model_type[0:2] == 'LS':# LSTM
      input_shape = (cg.par_num,) + cg.dim + (1,)
    elif model_type[0:2] == 'CN':  #CNN
      input_shape = cg.dim + (cg.par_num,)
    print('intput shape: ', input_shape)

    model_inputs = [Input(input_shape)]
    model_outputs=[]
    if model_type[0:2] == 'LS':
      tx, ty, theta = model_param.get_LSTM(input_shape ,dimension = 3,activate = True, batch_norm = True )(model_inputs[0])
    elif model_type[0:2] == 'CN':
      tx, ty, theta = model_param.get_CNN(nb_filters=[16,32,64,128,128])(model_inputs[0])
    model_outputs += [tx]
    model_outputs += [ty]
    model_outputs += [theta]
    model = Model(inputs = model_inputs,outputs = model_outputs)
    print(model.summary())

    model.load_weights(model_file)

    # predict generator
    print('Predict...') 

    for i in range(0,x_par_image_predict.shape[0]):

        # do both cross-validation and prediction
        # if batch_list[i] != (int(m[0])) and batch_list[i] != 5: # batch 5 is the testing cohort
        #     continue

        patient_subid = patient_subid_list[n[i]]
        patient_id = patient_id_list[n[i]]
        random_name = random_name_list[n[i]]
        start_slice = start_slice_list[n[i]]
        print(batch_list[i], patient_id,patient_subid,random_name,start_slice)

        datagen = Generator.DataGenerator(np.asarray([x_par_image_predict[i]]),np.asarray([y_motion_param_predict[i]]),
                                    model_type = model_type,
                                    patient_num = 1, 
                                    batch_size = cg.batch_size, 
                                    input_dimension = input_shape,
                                    output_dimension = (4,),
                                    shuffle = False,
                                    add_noise = False,
                                    noise_sigma = 0,
                                    seed = 10)

        tx, ty, theta = model.predict_generator(datagen, verbose = 1, steps = 1,)

        # load ground truth
        truth = np.load(y_motion_param_predict[i], allow_pickle = True)
        tx_true = truth[0,:][0][1:5]
        ty_true = truth[2,:][0][1:5]
        r_true= truth[5,:][0][1:5]; theta_true = truth[4,:][0][1:5]

        # predict (convert back to pixels)
        tx = ff.convert_translation_control_points(tx, cg.dim[0], from_pixel_to_1 = False)
        ty = ff.convert_translation_control_points(ty, cg.dim[1], from_pixel_to_1 = False)
        r = [i / np.pi * 180 for i in theta]

        print(tx_true, tx,  np.asarray(tx) * 2, ty_true,ty, np.asarray(ty) * 2, r_true, r)

        result = [m[0], batch_list[i],patient_id,patient_subid,random_name,start_slice, 
        tx_true[0], tx_true[1], tx_true[2],tx_true[3], tx[0][0], tx[0][1], tx[0][2],tx[0][3],np.mean(np.abs(np.asarray(tx_true) - tx[0])),
        ty_true[0], ty_true[1], ty_true[2],ty_true[3], ty[0][0], ty[0][1], ty[0][2],ty[0][3],np.mean(np.abs(np.asarray(ty_true) - ty[0])),
        r_true[0], r_true[1], r_true[2],r_true[3], r[0][0], r[0][1], r[0][2],r[0][3],np.mean(np.abs(np.asarray(r_true) - r[0]))]

#         # make corrected PAR
#         # true image
#         img_true = nb.load(motion_free_file_list[i]); affine =img_true.affine; img_true = img_true.get_fdata()
#         # motion image
#         img_motion = nb.load(motion_file_list[i]); img_motion = img_motion.get_fdata()
#         # find motion for each time frame
#         spline_x,_,_,_ = transform.spline_fit(np.linspace(0,1,5), np.asarray([0,tx[0][0],tx[0][1], tx[0][2], tx[0][3]])); tx_25 = [spline_x(l) for l in np.linspace(0.04, 1,25 )]
#         spline_y,_,_,_ = transform.spline_fit(np.linspace(0,1,5), np.asarray([0,ty[0][0],ty[0][1], ty[0][2], ty[0][3]])); ty_25 = [spline_y(l) for l in np.linspace(0.04, 1,25 )]
#         spline_theta,_,_,_ = transform.spline_fit(np.linspace(0,1,5), np.asarray([0,theta[0][0],theta[0][1], theta[0][2], theta[0][3]])); theta_25 = [spline_theta(l) for l in np.linspace(0.04, 1,25 )]

#         par1 = np.load(os.path.join(cg.data_dir, 'PAR_2D_spline',patient_id,patient_subid,random_name,'slice_0_to_15/original/PARs_original.npy'), allow_pickle = True)
#         par2 = np.load(os.path.join(cg.data_dir, 'PAR_2D_spline',patient_id,patient_subid,random_name,'slice_15_to_30/original/PARs_original.npy'), allow_pickle = True)
#         par3 = np.load(os.path.join(cg.data_dir, 'PAR_2D_spline',patient_id,patient_subid,random_name,'slice_30_to_45/original/PARs_original.npy'), allow_pickle = True)
#         par4 = np.load(os.path.join(cg.data_dir, 'PAR_2D_spline',patient_id,patient_subid,random_name,'slice_45_to_60/original/PARs_original.npy'), allow_pickle = True)
#         par = np.concatenate([par1,par2,par3,par4],axis = -1)
#         print(par.shape)
        
#         par_c = np.copy(par)
#         for j in range(0,par.shape[0]):
#             I = par[j,...]
#             _,_,_,transformation_matrix = transform.generate_transform_matrix([-tx_25[j], -ty_25[j],0],[0,0, -theta_25[j]],[1,1,1],I.shape)
#             transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, I.shape)
#             img_new = transform.apply_affine_transform(I, transformation_matrix,3, cval = np.min(par))
#             par_c[j,...] = img_new

#         par_c = np.sum(par_c,axis=0) / 25
#         ff.make_folder([os.path.join(save_folder,'images',patient_id), os.path.join(save_folder,'images',patient_id,patient_subid,
#                         os.path.join(save_folder,'images',patient_id,patient_subid, random_name), os.path.join(save_folder,'images',patient_id,patient_subid, random_name, 'batch_'+str(m[0])))])
#         nb.save(nb.Nifti1Image(par_c,affine), os.path.join(save_folder,'images',patient_id,patient_subid, random_name, 'batch_'+str(m[0]), 'pred.nii.gz') )

#         # use true motion to recon:
#         if os.path.isfile(os.path.join(save_folder,'images',patient_id,patient_subid, random_name, 'truth.nii.gz')) == 0:
#             spline_x,_,_,_ = transform.spline_fit(np.linspace(0,1,5), np.asarray([0,tx_true[0],tx_true[1], tx_true[2], tx_true[3]])); tx_25 = [spline_x(l) for l in np.linspace(0.04, 1,25 )]
#             spline_y,_,_,_ = transform.spline_fit(np.linspace(0,1,5), np.asarray([0,ty_true[0],ty_true[1], ty_true[2], ty_true[3]])); ty_25 = [spline_y(l) for l in np.linspace(0.04, 1,25 )]
#             spline_theta,_,_,_ = transform.spline_fit(np.linspace(0,1,5), np.asarray([0,theta_true[0],theta_true[1], theta_true[2], theta_true[3]])); theta_25 = [spline_theta(l) for l in np.linspace(0.04, 1,25 )]
#             par_c_t = np.copy(par)
#             for j in range(0,par.shape[0]):
#                 I = par[j,...]
#                 _,_,_,transformation_matrix = transform.generate_transform_matrix([-tx_25[j], -ty_25[j],0],[0,0, -theta_25[j]],[1,1,1],I.shape)
#                 transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, I.shape)
#                 img_new = transform.apply_affine_transform(I, transformation_matrix,3, cval = np.min(par))
#                 par_c_t[j,...] = img_new

#             par_c_t = np.sum(par_c_t,axis=0) / 25
#             nb.save(nb.Nifti1Image(par_c_t,affine), os.path.join(save_folder,'images',patient_id,patient_subid, random_name, 'truth.nii.gz') )
#             mae_truth,_,_,_,_,_ = ff.compare(par_c_t, img_true,-100)


#         # calculate image comparison
#         mae_motion,_,_,_,_,_ = ff.compare(img_motion,img_true,-100)
#         mae,_,_,_,_,_ = ff.compare(par_c, img_true,-100)
#         result = result + [mae, mae_motion, mae_truth]
#         print(mae,mae_motion, mae_truth)
#         Results.append(result)

    

# df = pd.DataFrame(Results, columns= ['which_model','batch','PatientID', 'AccessionNumber', 'MotionName', 'PAR_start_slice',
#                                     'tx_true','tx_true','tx_true','tx_true','tx','tx','tx','tx','tx_mae',
#                                     'ty_true','ty_true','ty_true','ty_true','ty','ty','ty','ty','ty_mae',
#                                     'r_true','r_true','r_true','r_true','r','r','r','r','r_mae',
#                                     'image_mae', 'raw_mae','true_mae'
#                                     ])
# df.to_excel(os.path.join(save_folder, 'parameters_validation.xlsx'), index = False)
    