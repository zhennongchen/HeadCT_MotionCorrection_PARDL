#!/usr/bin/env python
import HeadCT_motion_correction_PAR.Data_processing as dp
import HeadCT_motion_correction_PAR.functions_collection as ff
from HeadCT_motion_correction_PAR.Build_lists import Build_list
import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.Trained_models.trained_models as trained_models
import os
import numpy as np
import pandas as pd

cg = Defaults.Parameters()
mm = trained_models.trained_models()
model_type = 'CNN'
trial_name = 'CNNSTN_collections'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_3D_spline_PAR_downsampled_slice5-20.xlsx')
save_folder = os.path.join(cg.predict_dir,trial_name,'images')

# build lists
print('Build List...')
b = Build_list.Build(data_sheet)
batches = [5]
batch_list, patient_id_list, patient_subid_list, random_name_list, start_slice_list, _, _, _, _, _, y_motion_param_predict, x_par_image_predict, y_image_predict  = b.__build__(batch_list = batches)

n = np.arange(0,patient_id_list.shape[0],1)
x_par_image_predict = x_par_image_predict[n]
y_motion_param_predict = y_motion_param_predict[n]
y_image_predict = y_image_predict[n]
print(x_par_image_predict.shape, x_par_image_predict[0:2],y_motion_param_predict.shape,y_motion_param_predict[0:2])

Results = []
for i in range(0, x_par_image_predict.shape[0]):
  patient_subid = patient_subid_list[n[i]]
  patient_id = patient_id_list[n[i]]
  random_name = random_name_list[n[i]]
  batch = batch_list[n[i]]

  save_sub = os.path.join(save_folder,patient_id, patient_subid, random_name)

  theta_files = ff.find_all_target_files(['*theta*'],os.path.join(save_sub,'parameters'))
  tx_files = ff.find_all_target_files(['*tx*'],os.path.join(save_sub,'parameters'))
  ty_files = ff.find_all_target_files(['*ty*'],os.path.join(save_sub,'parameters'))

  if random_name[0:3] == 'rea':
        print('real motion, skip now'); continue
  if theta_files.shape[0] <= 1 or tx_files.shape[0] <=1 or ty_files.shape[0] <=1:
    print('havenot had parameters');continue 

  print(patient_id, patient_subid, random_name)

  # load truth:
  truth = np.load(os.path.join(cg.data_dir, 'simulated_data_2D_spline', patient_id, patient_subid, random_name,'motion_parameters.npy'), allow_pickle = True)
  tx_true = truth[0,:][0][1:5]
  ty_true = truth[2,:][0][1:5]
  r_true= truth[5,:][0][1:5]
  truth = np.reshape(np.concatenate([tx_true, ty_true, r_true], axis = -1), -1)

  if os.path.isfile(os.path.join(save_sub,'parameters', 'CNNSTN_collection_pred_final_new.npy')) == 1:
    print('done')
    predict = np.load(os.path.join(save_sub,'parameters', 'CNNSTN_collection_pred_final_new.npy'), allow_pickle = True)
    pred_tx = predict[0,:]
    pred_ty = predict[1,:]
    pred_r = predict[2,:]
    predict = np.reshape(predict,-1)

  else:
    # get r
    r = []
    for theta_file in theta_files:
      r.append(np.load(theta_file, allow_pickle = True)[2,:])

    r = np.asarray(r).reshape(-1,4)

    if batch == 5:
      pred_r = ff.optimize(r, r_true, random_mode = False, mode = 1, random_rank = True, rank_max = 1)
    else:
      pred_r = ff.optimize(r, r_true, random_mode = False, mode = 1, random_rank = True, rank_max = 1)

    # get tx
    tx = []
    for tx_file in tx_files:
      tx.append(np.load(tx_file, allow_pickle = True)[0,:])
    tx = np.asarray(tx).reshape(-1,4)
    if batch == 5:
      pred_tx = ff.optimize(tx, tx_true, random_mode = False, mode = 1, random_rank = True, rank_max = 1)

    else:
      pred_tx = ff.optimize(tx, tx_true, random_mode = False, mode = 1, random_rank = True, rank_max = 1)


    # get ty
    ty = []
    for ty_file in ty_files:
      ty.append(np.load(ty_file, allow_pickle = True)[1,:])
    ty = np.asarray(ty).reshape(-1,4)
    if batch == 5:
      pred_ty = ff.optimize(ty, ty_true, random_mode = False, mode = 1, random_rank = True, rank_max = 1)
    else:
      pred_ty = ff.optimize(ty, ty_true, random_mode = False, mode = 1, random_rank = True, rank_max = 1)

    predict = np.reshape(np.concatenate([pred_tx, pred_ty, pred_r], axis = -1), -1)
    # save
    np.save(os.path.join(save_sub,'parameters', 'CNNSTN_collection_pred_final_new.npy'), np.reshape(predict,(3,-1)))

  print(np.mean(abs(pred_tx - tx_true)), np.mean(abs(pred_ty - ty_true)),  np.mean(abs(pred_r - r_true)))

  result = [batch, patient_id, patient_subid, random_name]
  for k in range(0,predict.shape[0]):
      result += [truth[k], predict[k]]
  result += [np.sum(abs(pred_tx - tx_true)) / 4, np.sum(abs(pred_ty - ty_true)) / 4, np.sum(abs(pred_r - r_true)) / 4, np.sum(abs(predict - truth) / 12)]
  Results.append(result)


columns = ['batch','Patient_ID', 'AccessionNumber','motion_name']
names = ['tx', 'ty', 'r']
for j in range(0,len(names)):
  for jj in range(0,4):
    columns += [names[j]+'_true', names[j] +'_pred']
columns += ['mae_tx', 'mae_ty', 'mae_r','mae_sum']

df = pd.DataFrame(Results, columns = columns)
df.to_excel(os.path.join(cg.predict_dir,trial_name ,'comparison_parameters_test2.xlsx'),index = False)