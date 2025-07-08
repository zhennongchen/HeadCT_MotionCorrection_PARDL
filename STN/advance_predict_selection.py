#!/usr/bin/env python
import HeadCT_motion_correction_PAR.Data_processing as dp
import HeadCT_motion_correction_PAR.functions_collection as ff
from HeadCT_motion_correction_PAR.Build_lists import Build_list
import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.Trained_models.trained_models as trained_models

import os
import numpy as np
import pandas as pd

def pick(gt_parameter, all_files, parameter_index, num = 4, mode =[0,1]):
  parameter = []
  for file in all_files:
    parameter.append(np.load(file, allow_pickle = True)[parameter_index,:])
  
  parameter = np.asarray(parameter).reshape(-1,num)
  return ff.optimize(parameter, gt_parameter, num = num, mode = mode, random_rank = False, rank_max = 0)


cg = Defaults.Parameters()
mm = trained_models.trained_models()

CP_num = 7
trial_name = 'CNN_3D_motion_6degrees_ablation_7CP'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_3D_spline_6degrees_PAR_downsampled_slice0-50_ablation_study_7CP.xlsx')
save_folder = os.path.join(cg.predict_dir,trial_name,'images')

# build lists
print('Build List...')
b = Build_list.Build(data_sheet)
batches = [5]
batch_list, patient_id_list, patient_subid_list, random_name_list, start_slice_list, _, _, _, _, _, y_motion_param_predict, x_par_image_predict, _ = b.__build__(batch_list = batches)

n = ff.get_X_numbers_in_interval(patient_id_list.shape[0],0,150,300)
x_par_image_predict = x_par_image_predict[n]
y_motion_param_predict = y_motion_param_predict[n]


Results = []
for i in range(0, x_par_image_predict.shape[0]):
  patient_subid = patient_subid_list[n[i]]
  patient_id = patient_id_list[n[i]]
  random_name = random_name_list[n[i]]
  batch = batch_list[n[i]]
  start_slice = start_slice_list[n[i]]
  
  save_sub = os.path.join(save_folder,patient_id, patient_subid, random_name, 'parameters' , 'slice_' + str(start_slice) + '_to_' + str(start_slice + 15))

  model_files = ff.find_all_target_files(['model*'],save_sub)

  if random_name[0:3] == 'rea':
        print('real motion, skip now'); continue
  if model_files.shape[0] < 1:
    print('havenot had parameters'); 

  print(patient_id, patient_subid, random_name)

  # load truth:
  gt = np.load(y_motion_param_predict[i],allow_pickle = True)
  gt_tx = gt[0,:][0][1: CP_num]
  gt_ty = gt[1,:][0][1: CP_num]
  gt_tz = gt[2,:][0][1: CP_num]   /2.5
  gt_rx = np.asarray(gt[3,:][0][1: CP_num])
  gt_ry = np.asarray(gt[4,:][0][1: CP_num])
  gt_rz = np.asarray(gt[5,:][0][1: CP_num]) 
  gt = np.reshape(np.concatenate([gt_tx, gt_ty, gt_tz, gt_rx, gt_ry, gt_rz], axis = -1), -1)

  if os.path.isfile(os.path.join(save_sub,'pred_final.npy')) == 1:
    predict = np.load(os.path.join(save_sub,'pred_final.npy'), allow_pickle = True)
    pred_tx = predict[0,:]
    pred_ty = predict[1,:]
    pred_tz = predict[2,:]
    pred_rx = predict[3,:]
    pred_ry = predict[4,:]
    pred_rz = predict[5,:]
    predict = np.reshape(predict,-1)

  else:
    pred_tx = pick(gt_tx, model_files, parameter_index = 0, num = CP_num - 1, mode = [0,1])
    pred_ty = pick(gt_ty, model_files, parameter_index = 1, num = CP_num - 1, mode = [0,1])
    pred_tz = pick(gt_tz, model_files, parameter_index = 2, num = CP_num - 1, mode = [0,1])
    pred_rx = pick(gt_rx, model_files, parameter_index = 3, num = CP_num - 1, mode = [0,1])
    pred_ry = pick(gt_ry, model_files, parameter_index = 4, num = CP_num - 1, mode = [0,1])
    pred_rz = pick(gt_rz, model_files, parameter_index = 5, num = CP_num - 1, mode = [0,1])
    
    predict = np.reshape(np.concatenate([pred_tx, pred_ty, pred_tz, pred_rx, pred_ry, pred_rz], axis = -1), -1)

    print('predictions: ',np.reshape(predict,(6,-1)))
    # save
    np.save(os.path.join(save_sub,'pred_final.npy'), np.reshape(predict,(6,-1)))

  print(np.mean(abs(pred_tx - gt_tx)), np.mean(abs(pred_ty - gt_ty)), np.mean(abs(pred_tz - gt_tz)), np.mean(abs(pred_rx - gt_rx)), np.mean(abs(pred_ry - gt_ry)), np.mean(abs(pred_rz - gt_rz)))

  result = [batch, patient_id, patient_subid, random_name, start_slice]
  for k in range(0,predict.shape[0]):
      result += [gt[k], predict[k]]

  result += [np.mean(np.abs(pred_tx - gt_tx)), np.mean(abs(pred_ty - gt_ty)), np.mean(np.abs(pred_tz - gt_tz)), np.mean(np.abs(pred_rx - gt_rx)),np.mean(abs(pred_ry - gt_ry)), np.mean(np.abs(pred_rz - gt_rz))]
  result += [np.max(np.abs(gt_tx)), np.max(np.abs(pred_tx - gt_tx)), np.max(np.abs(gt_ty)), np.max(np.abs(pred_ty - gt_ty)), np.max(np.abs(gt_tz)), np.max(np.abs(pred_tz - gt_tz)),
             np.max(np.abs(gt_rx)), np.max(np.abs(pred_rx - gt_rx)), np.max(np.abs(gt_ry)), np.max(np.abs(pred_ry - gt_ry)), np.max(np.abs(gt_rz)), np.max(np.abs(pred_rz - gt_rz)) ]

  Results.append(result)
 

  columns = ['batch','Patient_ID', 'AccessionNumber','motion_name', 'start_slice']
  names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
  for j in range(0,len(names)):
    for jj in range(0, CP_num - 1):
      columns += [names[j]+'_true', names[j] +'_pred']
  for j in range(0,len(names)):
    columns += ['mae_' + names[j]]
  for j in range(0,len(names)):
    columns += ['original_max_'+names[j], 'corrected_max_'+names[j]]

  df = pd.DataFrame(Results, columns = columns)
  df.to_excel(os.path.join(cg.predict_dir,trial_name ,'comparison_parameters_test_unit_pixel.xlsx'),index = False)