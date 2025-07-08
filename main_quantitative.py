#!/usr/bin/env python
import HeadCT_motion_correction_PAR.functions_collection as ff
from HeadCT_motion_correction_PAR.Build_lists import Build_list
import HeadCT_motion_correction_PAR.Defaults as Defaults

import os
import numpy as np
import pandas as pd
import nibabel as nb

cg = Defaults.Parameters()
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_3D_spline_6degrees_PAR_downsampled_slice0-50.xlsx')

print('Build List...')
b = Build_list.Build(data_sheet)
batches = [5]
batch_list, patient_id_list, patient_subid_list, random_name_list, start_slice_list, _, _, _, _, _, y_motion_param_predict, x_par_image_predict, _ = b.__build__(batch_list = batches)

n = np.arange(0,patient_id_list.shape[0], 3)
m = ff.get_X_numbers_in_interval(n.shape[0],0,20, 100)
n = n[m]


Results = []

for i in range(0,n.shape[0]):

    patient_subid = patient_subid_list[n[i]]
    patient_id = patient_id_list[n[i]]
    random_name = random_name_list[n[i]]
    print(patient_id, patient_subid, random_name)

    r = [patient_id, patient_subid, random_name]

    # ground truth static
    static = nb.load(os.path.join(cg.data_dir, 'simulated_data_3D_spline', patient_id, patient_subid, 'static', 'image_data/recon_partial.nii.gz')).get_fdata()
    
    # motion:
    motion = nb.load(os.path.join(cg.data_dir, 'simulated_data_3D_spline_6degrees', patient_id, patient_subid, random_name, 'image_data/recon_partial.nii.gz')).get_fdata()

    # PAR method:
    static_HR = nb.load(os.path.join(cg.data_dir,'raw_data/nii-images/thin_slice',patient_id, patient_subid, 'img-nii-0.625/img.nii.gz')).get_fdata()
    par_HR = nb.load(os.path.join(cg.predict_dir,  'CNN_3D_motion_6degrees/images', patient_id, patient_subid, random_name, 'pred_PAR_corrected_HR_z.nii.gz')).get_fdata()
    par = np.zeros([par_HR.shape[0], par_HR.shape[1], 50])
    for zz in range(0,50):
        par[:,:,zz] = np.mean(par_HR[:,:,  (40 + zz * 4) : (40 + (zz+1) * 4) ], -1)

    static_average = np.zeros([static_HR.shape[0], static_HR.shape[1], 50])
    for zz in range(0,50):
        static_average[:,:,zz] = np.mean(static_HR[:,:,  (40 + zz * 4) : (40 + (zz+1) * 4) ], -1)

    if os.path.isfile(os.path.join(cg.predict_dir,  'CNN_3D_motion_6degrees/images', patient_id, patient_subid, random_name, 'pred_PAR_corrected_HR_z_slice35.nii.gz')) == 1:
        print('use top')
        par_HR_top = nb.load(os.path.join(cg.predict_dir,  'CNN_3D_motion_6degrees/images', patient_id, patient_subid, random_name, 'pred_PAR_corrected_HR_z_slice35.nii.gz')).get_fdata()
        par_top = np.zeros([par_HR_top.shape[0], par_HR_top.shape[1], 50])
        for zz in range(0,50):
            par_top[:,:,zz] = np.mean(par_HR_top[:,:,  (40 + zz * 4) : (40 + (zz+1) * 4) ], -1)
    else:
        par_top = np.copy(par)

    # CNN method
    cnn = nb.load(os.path.join(cg.predict_dir, 'CNN_DResNet_patch_new/6degrees/images', patient_id, patient_subid, random_name, 'pred_CNN.nii.gz')).get_fdata()

    # static vs motion:
    mae_motion_all, _, rmse_motion_all, _, ssim_motion_all = ff.compare(motion[:,:,5:45], static[:,:,5:45], cutoff_low=-100)
    mae_motion_bottom, _, rmse_motion_bottom, _, ssim_motion_bottom = ff.compare(motion[:,:,5:20], static[:,:,5:20], cutoff_low=-100)
    mae_motion_middle, _, rmse_motion_middle, _, ssim_motion_middle = ff.compare(motion[:,:,20:35], static[:,:,20:35], cutoff_low=-100)
    mae_motion_top, _, rmse_motion_top, _, ssim_motion_top = ff.compare(motion[:,:,35:45], static[:,:,35:45], cutoff_low=-100)

    r  =  r + [mae_motion_all, rmse_motion_all, ssim_motion_all,mae_motion_bottom, rmse_motion_bottom, ssim_motion_bottom,
              mae_motion_middle, rmse_motion_middle, ssim_motion_middle, mae_motion_top, rmse_motion_top, ssim_motion_top ]

    # static vs cnn:
    mae_cnn_all, _, rmse_cnn_all, _, ssim_cnn_all = ff.compare(cnn[:,:,5:45], static[:,:,5:45], cutoff_low=-100)
    mae_cnn_bottom, _, rmse_cnn_bottom, _, ssim_cnn_bottom = ff.compare(cnn[:,:,5:20], static[:,:,5:20], cutoff_low=-100)
    mae_cnn_middle, _, rmse_cnn_middle, _, ssim_cnn_middle = ff.compare(cnn[:,:,20:35], static[:,:,20:35], cutoff_low=-100)
    mae_cnn_top, _, rmse_cnn_top, _, ssim_cnn_top = ff.compare(cnn[:,:,35:45], static[:,:,35:45], cutoff_low=-100)

    r  =  r + [mae_cnn_all, rmse_cnn_all, ssim_cnn_all,mae_cnn_bottom, rmse_cnn_bottom, ssim_cnn_bottom,
              mae_cnn_middle, rmse_cnn_middle, ssim_cnn_middle, mae_cnn_top, rmse_cnn_top, ssim_cnn_top ]


    # static vs. par:
    mae_par_bottom, _, rmse_par_bottom, _, ssim_par_bottom = ff.compare(par[:,:,5:20], static_average[:,:,5:20], cutoff_low=-100, extreme = 900)
    mae_par_middle, _, rmse_par_middle, _, ssim_par_middle = ff.compare(par[:,:,20:35], static_average[:,:,20:35], cutoff_low=-100, extreme = 900)
    mae_par_top, _, rmse_par_top, _, ssim_par_top = ff.compare(par_top[:,:,35:45], static_average[:,:,35:45], cutoff_low=-100, extreme = 900)
    mae_par_all = (mae_par_bottom + mae_par_middle + mae_par_top) / 3
    rmse_par_all = (rmse_par_bottom + rmse_par_middle + rmse_par_top) / 3
    ssim_par_all = (ssim_par_bottom + ssim_par_middle + ssim_par_top) / 3

    r  =  r + [mae_par_all, rmse_par_all, ssim_par_all,mae_par_bottom, rmse_par_bottom, ssim_par_bottom,
              mae_par_middle, rmse_par_middle, ssim_par_middle, mae_par_top, rmse_par_top, ssim_par_top ]


    Results.append(r)

    column_list = ['Patient_ID', 'AccessionNumber', 'motion_name']
    classes  = ['motion', 'CNN', 'PAR']
    errrors = ['mae', 'rmse', 'ssim']
    segment = ['all', 'bottom', 'middle', 'top']
    for c in classes:
        for s in segment:
            for e in errrors:
                column_list  = column_list + [c +'_' + s + '_' + e]
    

    df = pd.DataFrame(Results, columns = column_list)
    df.to_excel(os.path.join(cg.predict_dir,'comparison_list2.xlsx'), index = False) 
    


   
