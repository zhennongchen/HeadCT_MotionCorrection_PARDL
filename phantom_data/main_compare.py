import HeadCT_motion_correction_PAR.functions_collection as ff
from HeadCT_motion_correction_PAR.Build_lists import Build_list
import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.Trained_models.trained_models as trained_models
import HeadCT_motion_correction_PAR.motion_simulator.transformation as transform
import os
import numpy as np
import nibabel as nb
import pandas as pd


cg = Defaults.Parameters()
mm = trained_models.trained_models()
trial_name = 'phantom_data'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_phantom_data.xlsx')
save_folder = os.path.join(cg.predict_dir,trial_name,'images')
ff.make_folder([os.path.dirname(save_folder), save_folder])

# build lists
print('Build List...')
b = Build_list.Build(data_sheet)
batches = [5]
batch_list, patient_id_list, patient_subid_list, random_name_list, start_slice_list, end_slice_list, _, _, _, _, y_motion_param_predict, x_par_image_predict, _ = b.__build__(batch_list = batches)

n = np.arange(0,21,1)
y_motion_param_predict = y_motion_param_predict[n]
patient_id_list = patient_id_list[n]


# compare parameters:
# parameter_results = []
# for i in range(0, patient_id_list.shape[0]):
#     study = patient_id_list[i]
#     print(study)

#     # compare the motion estimation
#     motion_gt = np.load(os.path.join(cg.data_dir, 'phantom_data', study , 'motion_each_point.npy'),allow_pickle = True)
#     motion_pred = np.load(os.path.join(cg.predict_dir, 'phantom_data/images', study, 'parameters/pred_final.npy'), allow_pickle = True)

#     spline_tx_pred = transform.interp_func(np.linspace(0,100,5), np.concatenate([np.asarray([0]),motion_pred[0,:]],axis = -1))
#     spline_tz_pred = transform.interp_func(np.linspace(0,100,5), np.concatenate([np.asarray([0]),motion_pred[1,:]],axis = -1))
#     spline_rx_pred = transform.interp_func(np.linspace(0,100,5), np.concatenate([np.asarray([0]),motion_pred[2,:]],axis = -1))
#     spline_rz_pred = transform.interp_func(np.linspace(0,100,5), np.concatenate([np.asarray([0]),motion_pred[3,:]],axis = -1))

#     pred_tx = spline_tx_pred(np.linspace(0,100,25))
#     pred_tz = spline_tz_pred(np.linspace(0,100,25))
#     pred_rx = spline_rx_pred(np.linspace(0,100,25))
#     pred_rz = spline_rz_pred(np.linspace(0,100,25))

#     gt_tx = np.asarray(motion_gt[0,:])
#     gt_tz = np.asarray(motion_gt[2,:])
#     gt_rx = np.asarray(motion_gt[3,:])
#     gt_rz = np.asarray(motion_gt[5,:])
   

#     print(np.mean(abs(pred_tx - gt_tx)), np.mean(abs(pred_tz - gt_tz)), np.mean(abs(pred_rx - gt_rx)), np.mean(abs(pred_rz - gt_rz)))

#     parameter_result = [study]

#     parameter_result += [np.mean(np.abs(pred_tx - gt_tx)), np.mean(np.abs(pred_tz - gt_tz)), np.mean(np.abs(pred_rx - gt_rx)), np.mean(np.abs(pred_rz - gt_rz))]
    
#     parameter_results.append(parameter_result)
    
#     columns = ['Patient_ID']
#     names = ['tx', 'tz', 'rx', 'rz']
#     for j in range(0,len(names)):
#       columns += ['mae_' + names[j]]

#     df = pd.DataFrame(parameter_results, columns = columns)
#     df.to_excel(os.path.join(cg.predict_dir,trial_name ,'comparison_parameters.xlsx'),index = False)



# compare image
image_results = []
for i in range(0, patient_id_list.shape[0]):
    study = patient_id_list[i]
    print(study)

    par_results = nb.load(os.path.join(cg.data_dir,'phantom_data', study, 'PAR_corrected_ibc_idr.nii.gz')).get_fdata()

    cnn_results = nb.load(os.path.join(cg.predict_dir, 'phantom_data/images/',study, 'pred_CNN_new.nii.gz')).get_fdata()

    static = nb.load('/mnt/camca_NAS/head_phantom_raw/processed/study_4/scan4/processed_0/recon_cropped/recon_ibc_idr_partial.nii.gz').get_fdata()

    move = nb.load(os.path.join(cg.data_dir,'phantom_data', study, 'simulated_imgs_resampled/recon_motion_ibc_idr_partial.nii.gz')).get_fdata()

    mae_move, mse, rmse_move, r_rmse, ssim_move = ff.compare(move, static, cutoff_low=-10)
    print(mae_move, rmse_move, ssim_move)

    mae_cnn, mse, rmse_cnn, r_rmse, ssim_cnn = ff.compare(cnn_results, static, cutoff_low=-10)
    print(mae_cnn, rmse_cnn, ssim_cnn)

    mae_par, mse, rmse_par, r_rmse, ssim_par = ff.compare(par_results, static, cutoff_low=-10, extreme = 900)
    print(mae_par, rmse_par, ssim_par)

    image_result = [study]
    image_result += [mae_move, rmse_move, ssim_move,mae_cnn, rmse_cnn, ssim_cnn , mae_par, rmse_par, ssim_par]
    image_results.append(image_result)

    columns = ['Patient_ID', 'mae_motion' ,'rmse_motion', 'ssim_motion', 'mae_cnn', 'rmse_cnn', 'ssim_cnn', 'mae_par', 'rmse_par', 'ssim_par']
    
    
    df = pd.DataFrame(image_results, columns = columns)
    df.to_excel(os.path.join(cg.predict_dir,trial_name ,'comparison_images_new.xlsx'),index = False)
