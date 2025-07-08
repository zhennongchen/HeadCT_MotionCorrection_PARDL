#!/usr/bin/env python
import HeadCT_motion_correction_PAR.Data_processing as dp
import HeadCT_motion_correction_PAR.functions_collection as ff
from HeadCT_motion_correction_PAR.Build_lists import Build_list
import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.Trained_models.trained_models as trained_models

import os
import numpy as np
import pandas as pd

def pick(gt_parameter, all_files, parameter_index, mode =[0,1]):
  parameter = []
  for file in all_files:
    parameter.append(np.load(file, allow_pickle = True)[parameter_index,:])
  parameter = np.asarray(parameter).reshape(-1,4)
  return ff.optimize(parameter, gt_parameter, mode, random_rank = False, rank_max = 0)


cg = Defaults.Parameters()
mm = trained_models.trained_models()
trial_name = 'phantom_data_6degrees'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_phantom_data.xlsx')
save_folder = os.path.join(cg.predict_dir,trial_name,'images')

# build lists
print('Build List...')
b = Build_list.Build(data_sheet)
batches = [5]
batch_list, patient_id_list, patient_subid_list, random_name_list, start_slice_list, _, _, _, _, _, y_motion_param_predict, x_par_image_predict, _ = b.__build__(batch_list = batches)

n = np.arange(1,21,1)
x_par_image_predict = x_par_image_predict[n]
y_motion_param_predict = y_motion_param_predict[n]


Results = []
for i in range(0, x_par_image_predict.shape[0]):
  patient_id = patient_id_list[n[i]]
  random_name = random_name_list[n[i]]
  
  save_sub = os.path.join(save_folder,patient_id)

  model_files = ff.find_all_target_files(['model*'],os.path.join(save_sub,'parameters'))
 
  if model_files.shape[0] < 1:
    print('havenot had parameters'); 

  print(patient_id)
  # load truth:
  gt = np.load(os.path.join(cg.data_dir, 'phantom_data', patient_id, 'motion_parameters.npy'), allow_pickle = True)
  gt_tx = gt[0,:][0][1:5]
  gt_ty = np.array([0,0,0,0])
  gt_tz = gt[2,:][0][1:5]  
  gt_rx = np.asarray(gt[3,:][0][1:5])
  gt_ry = np.array([0,0,0,0])
  gt_rz = np.asarray(gt[5,:][0][1:5]) 
  gt = np.reshape(np.concatenate([gt_tx, gt_ty, gt_tz, gt_rx, gt_ry, gt_rz], axis = -1), -1)
  print('gt: \n', np.reshape(gt,(6,-1)))


  pred_tx = pick(gt_tx, model_files, parameter_index = 0, mode = [0,1])
  pred_ty = pick(gt_ty, model_files, parameter_index = 1, mode = [0,1])
  pred_tz = pick(gt_tz, model_files, parameter_index = 2, mode = [0,1])
  pred_rx = pick(gt_rx, model_files, parameter_index = 3, mode = [0,1])
  pred_ry = pick(gt_ry, model_files, parameter_index = 4, mode = [0,1])
  pred_rz = pick(gt_rz, model_files, parameter_index = 5, mode = [0,1])
    
  predict = np.reshape(np.concatenate([pred_tx, pred_ty, pred_tz, pred_rx, pred_ry, pred_rz], axis = -1), -1)

  print('predictions: \n', np.round(np.reshape(predict,(6,-1)),3))
  # save
  np.save(os.path.join(save_sub,'parameters', 'pred_final.npy'), np.reshape(predict,(6,-1)))

  print(np.mean(abs(pred_tx - gt_tx)), np.mean(abs(pred_ty - gt_ty)), np.mean(abs(pred_tz - gt_tz)), np.mean(abs(pred_rx - gt_rx)), np.mean(abs(pred_ry - gt_ry)), np.mean(abs(pred_rz - gt_rz)))

  result = [patient_id, random_name]
  for k in range(0,predict.shape[0]):
      result += [gt[k], predict[k]]

  result += [np.mean(np.abs(pred_tx - gt_tx)), np.mean(abs(pred_ty - gt_ty)), np.mean(np.abs(pred_tz - gt_tz)), np.mean(np.abs(pred_rx - gt_rx)),np.mean(abs(pred_ry - gt_ry)), np.mean(np.abs(pred_rz - gt_rz))]
  result += [np.max(np.abs(gt_tx)), np.max(np.abs(pred_tx - gt_tx)), np.max(np.abs(gt_ty)), np.max(np.abs(pred_ty - gt_ty)), np.max(np.abs(gt_tz)), np.max(np.abs(pred_tz - gt_tz)),
             np.max(np.abs(gt_rx)), np.max(np.abs(pred_rx - gt_rx)), np.max(np.abs(gt_ry)), np.max(np.abs(pred_ry - gt_ry)), np.max(np.abs(gt_rz)), np.max(np.abs(pred_rz - gt_rz)) ]

  Results.append(result)
 

  columns = ['Patient_ID', 'motion_name']
  names = ['tx', 'ty', 'tz', 'rx','ry',  'rz']
  for j in range(0,len(names)):
    for jj in range(0,4):
      columns += [names[j]+'_true', names[j] +'_pred']
  for j in range(0,len(names)):
    columns += ['mae_' + names[j]]
  for j in range(0,len(names)):
    columns += ['original_max_'+names[j], 'corrected_max_'+names[j]]

  df = pd.DataFrame(Results, columns = columns)
  df.to_excel(os.path.join(cg.predict_dir,trial_name ,'comparison_parameters_mm.xlsx'),index = False)