#!/usr/bin/env python
import model_STN
import Generator_STN
import HeadCT_motion_correction_PAR.Data_processing as dp
import HeadCT_motion_correction_PAR.functions_collection as ff
from HeadCT_motion_correction_PAR.Build_lists import Build_list
import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.Trained_models.trained_models as trained_models

import os
import numpy as np
import nibabel as nb
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input

cg = Defaults.Parameters()
mm = trained_models.trained_models()

CP_num = 6
trial_name = 'CNN_3D_motion_6degrees_ablation_6CP'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_3D_spline_6degrees_PAR_downsampled_slice0-50_ablation_study_6CP.xlsx')
save_folder = os.path.join(cg.predict_dir,trial_name,'images')
ff.make_folder([os.path.dirname(save_folder), save_folder])


# build lists
print('Build List...')
b = Build_list.Build(data_sheet)
batches = [5]
batch_list, patient_id_list, patient_subid_list, random_name_list, start_slice_list, end_slice_list, _, _, _, _, y_motion_param_predict, x_par_image_predict, _ = b.__build__(batch_list = batches)

n = ff.get_X_numbers_in_interval(patient_id_list.shape[0],0,150,300)
x_par_image_predict = x_par_image_predict[n]
y_motion_param_predict = y_motion_param_predict[n]
start_slice_list = start_slice_list[n]
print(x_par_image_predict.shape, x_par_image_predict[0:2],y_motion_param_predict.shape,y_motion_param_predict[0:2])

# create model architecture:
input_shape = cg.dim + (cg.par_num,)
model_inputs = [Input(input_shape)]
model_outputs=[]
tx, ty, tz, rx, ry, rz = model_STN.get_CNN(nb_filters = [16,32,64,128,256], dimension = 3, crop_STN = True, crop_size = 30, CP_num = CP_num)(model_inputs[0])
model_outputs = [tx, ty, tz, rx , ry, rz]

# define model list:
# tx_model_files, tz_model_files, rx_model_files, rz_model_files = mm.CNN_3D_motion_thin_slice()
# files = [tx_model_files, tz_model_files, rx_model_files, rz_model_files]
model_files = mm.CNN_3D_motion_6degrees_ablation_6CP()
files = [model_files]


# do prediction
for jj in range(0,4):
  for j in range(0,len(files)): # tx: j=0, tz: j = 1, rx: j=2, rz: j = 3
    f = files[j][jj]
    print('loading model: ', f, '\n\n')
    model= Model(inputs = model_inputs,outputs = model_outputs)
    model.load_weights(f)
    
    for i in range(0,x_par_image_predict.shape[0]):
      patient_subid = patient_subid_list[n[i]]
      patient_id = patient_id_list[n[i]]
      random_name = random_name_list[n[i]]
      batch = batch_list[n[i]]
      
      # real motion?
      if random_name[0:3] == 'rea':
        print('real motion, skip now'); continue

      print(batch, patient_id, patient_subid, random_name)

      save_sub = os.path.join(save_folder, patient_id, patient_subid, random_name,'parameters' , 'slice_' + str(start_slice_list[i]) + '_to_' + str(start_slice_list[i] + 15))
      ff.make_folder([os.path.join(save_folder,patient_id), os.path.join(save_folder,patient_id, patient_subid), os.path.join(save_folder, patient_id, patient_subid, random_name), os.path.dirname(save_sub), save_sub])
      # if j == 0:
      #   filename = 'tx_pixel_' + str(jj) +'.npy'
      # elif j == 1:
      #   filename = 'tz_pixel_' + str(jj) +'.npy'
      # elif j ==2:
      #   filename = 'rx_degree_' + str(jj)+ '.npy'
      # else:
      #   filename = 'rz_degree_' + str(jj)+ '.npy'
      filename = 'model_' + str(jj) + '.npy'

      # done?
      if os.path.isfile(os.path.join(save_sub,filename)) == 1:
        print('already done'); continue

      datagen = Generator_STN.DataGenerator(np.asarray([x_par_image_predict[i]]),
                                            np.asarray([y_motion_param_predict[i]]),
                                            np.asarray([start_slice_list[i]]),
                                            np.asarray([end_slice_list[i]]),
                                            start_slice_sampling = None,
                                            patient_num = 1, 
                                            batch_size = cg.batch_size, 
                                            input_dimension = input_shape,
                                            output_vector_dimension = (CP_num - 1,),
                                            shuffle = False,augment = False,)

      tx, ty, tz, rx, ry, rz = model.predict_generator(datagen, verbose = 1, steps = 1,)
      tx = np.reshape(np.asarray(tx),-1) *5
      ty = np.reshape(np.asarray(ty),-1) *5
      tz = np.reshape(np.asarray(tz), -1) *2  # *5 for thin slice, *2 for 2.5mm
      rx = np.reshape(np.asarray(rx),-1)  *5
      ry = np.reshape(np.asarray(ry),-1)  *5
      rz = np.reshape(np.asarray(rz),-1)  *5
      predict = np.reshape(np.concatenate([tx, ty, tz, rx,ry, rz], axis = -1), -1)

      # load ground truth
      gt = np.load(y_motion_param_predict[i],allow_pickle = True)
      gt_tx = gt[0,:][0][1: CP_num]
      gt_ty = gt[1,:][0][1: CP_num]
      gt_tz = gt[2,:][0][1: CP_num]  / 2.5
      gt_rx = np.asarray(gt[3,:][0][1: CP_num])
      gt_ry= np.asarray(gt[4,:][0][1: CP_num])
      gt_rz = np.asarray(gt[5,:][0][1: CP_num])  


      print('tx: ', tx, ' gt tx:', gt_tx,  ' tx diff: ', np.abs(tx - gt_tx), ' in average: ', np.mean(np.abs(tx - gt_tx)), 'origin max: ',np.max(np.abs(gt_tx)), ' now max: ', np.max(np.abs(tx - gt_tx)))
      print('ty: ', ty, ' gt ty:', gt_ty,  ' ty diff: ', np.abs(ty - gt_ty), ' in average: ', np.mean(np.abs(ty - gt_ty)), 'origin max: ',np.max(np.abs(gt_ty)), ' now max: ', np.max(np.abs(ty - gt_ty)))
      print('tz: ', tz, ' gt tz:', gt_tz,  ' tz diff: ', np.abs(tz - gt_tz), ' in average: ', np.mean(np.abs(tz - gt_tz)), 'origin max: ',np.max(np.abs(gt_tz)), ' now max: ', np.max(np.abs(tz - gt_tz)))
      print('rx: ', rx, ' gt rx:', gt_rx,  ' rx diff: ', np.abs(rx - gt_rx), ' in average: ', np.mean(np.abs(rx - gt_rx)), 'origin max: ',np.max(np.abs(gt_rx)), ' now max: ', np.max(np.abs(rx - gt_rx)))
      print('ry: ', ry, ' gt ry:', gt_ry,  ' ry diff: ', np.abs(ry - gt_ry), ' in average: ', np.mean(np.abs(ry - gt_ry)), 'origin max: ',np.max(np.abs(gt_ry)), ' now max: ', np.max(np.abs(ry - gt_ry)))
      print('rz: ', rz, ' gt rz:', gt_rz,  ' rz diff: ', np.abs(rz - gt_rz), ' in average: ', np.mean(np.abs(rz - gt_rz)), 'origin max: ',np.max(np.abs(gt_rz)), ' now max: ', np.max(np.abs(rz - gt_rz)))
      
      # svae

      np.save(os.path.join(save_sub,filename), np.reshape(predict,(6,-1)))
  
    