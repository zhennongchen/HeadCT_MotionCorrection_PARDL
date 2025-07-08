#!/usr/bin/env python

# this is the script for new data collected at 2024/04
# %%
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import glob as gb
import nibabel as nb 
import math
import pandas as pd
import os
from skimage.measure import block_reduce
import ct_basic as ct
import HeadCT_motion_correction_PAR.functions_collection as ff
import HeadCT_motion_correction_PAR.motion_simulator.transformation as transform
import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.Data_processing as dp


import ct_projector.projector.cupy as ct_projector

main_folder = '/mnt/camca_NAS/Portable_CT_data'

motion_type = 'simulated_all_motion_v2' # each gantry rotation has motion except the first one
amplitude_max_severe = 10 #8
displacement_max_severe = 6
amplitude_max_mild = 5 #5
displacement_max_mild = 3

motion_freq = 0.5 # 0.3 or 0.35


# define the patient list
patient_sheet = pd.read_excel(os.path.join(main_folder,'Patient_list', 'NEW_CT_concise_collected_fixed_static_edited.xlsx'),dtype={'Patient_ID': str, 'Patient_subID': str})
patient_sheet['use'] = patient_sheet['use'].fillna(0)
patient_sheet = patient_sheet[(patient_sheet['use'] != 0) & (patient_sheet['use'] != 'no')]
print('patient sheet len: ', len(patient_sheet))

data_folder = os.path.join(main_folder, 'nii_imgs_202404', 'static')
save_folder = os.path.join(main_folder, 'simulations_202404', motion_type)
ff.make_folder([save_folder])

# define patient list index and simulation index
L = np.arange(1,6)
patient_index_list = np.arange(0,100)

for i in patient_index_list:
    row = patient_sheet.iloc[i]
    patient_id = row['Patient_ID']
    patient_subid = row['Patient_subID']
    print('\n',i, patient_id, patient_subid)

    save_folder_patient = os.path.join(save_folder, patient_id, patient_subid)
    ff.make_folder([os.path.join(save_folder, patient_id), save_folder_patient])

    img_file = ff.find_all_target_files(['fixed/img_1mm.nii.gz'],os.path.join(data_folder, patient_id, patient_subid))
    if len(img_file) != 1:
        ValueError('no raw data')
    
    img,spacing,img_affine = ct.basic_image_processing(img_file[0])
    print('nib image shape: ',img.shape, ' spacing: ',spacing)

    # define projectors
    img = img[np.newaxis, ...]

    # load the static image as reference
    if os.path.isfile(os.path.join(main_folder, 'simulations_202404','simulated_all_motion_v1',patient_id, patient_subid, 'static','image_data','recon.nii.gz')) == 0:
        valueError('wrong static image path')
    else:
        static_ref= nb.load(os.path.join(main_folder, 'simulations_202404','simulated_all_motion_v1',patient_id, patient_subid, 'static','image_data','recon.nii.gz')).get_fdata()
        static_ref = np.rollaxis(static_ref,2,0)
        print('static ref shape: ', static_ref.shape)
    print('static_ref is None? ', static_ref is None)

    for random_i in L:

        # create folder
        if random_i == 0:
            random_folder = os.path.join(save_folder_patient,'static')
        else:
            random_folder = os.path.join(save_folder_patient,'random_' +str(random_i))
        ff.make_folder([random_folder, os.path.join(random_folder,'image_data')])
        print('\n',random_i  , 'random')

        assert os.path.isfile(os.path.join(main_folder, 'simulations_202404','simulated_all_motion_v3',patient_id, patient_subid, 'random_' +str(random_i),'image_data','recon.nii.gz')) == 1
        recon_previous = nb.load(os.path.join(main_folder, 'simulations_202404','simulated_all_motion_v3',patient_id, patient_subid, 'random_' +str(random_i),'image_data','recon.nii.gz')).get_fdata()
        recon_previous = np.rollaxis(recon_previous,2,0)

        # gantry coverage = 1cm, which means it rotates once for every 1cm interval in the z-axis
        # second, find out how many rotations we need
        print('spacing: ', spacing, ' img shape: ', img.shape)
        rotation_num = int(spacing[0] * img.shape[1] // 10)
        slice_coverage = 10
        if random_i == 0:
            rotation_num = 1
            slice_coverage = int(spacing[0] * img.shape[1] // 10) * 10
        print('rotation_num: ', rotation_num, ' slice_coverage: ', slice_coverage)

        # start to generate the motion for each rotation
        # set whether static or motion for each rotation
        if random_i != 0:
            while True:
                motion_status = [np.random.uniform(0,1) <motion_freq for k in range(rotation_num-2)]
                if motion_freq <=0:
                    break
                if np.sum(motion_status) > 0:
                    break
            motion_status = [False] + motion_status + [False] # add two static rotations at the beginning and end
        if random_i == 0: # static
            motion_status = [False] 
        print('motion_status: ', motion_status)
        
      
        recon = np.zeros([slice_coverage * rotation_num,img.shape[2],img.shape[3]])

        for rot_n in range(0,rotation_num):
            motion_type = motion_status[rot_n]

            if motion_status[rot_n] == False:
               recon[slice_coverage * (rot_n) : slice_coverage * (rot_n + 1),...] = static_ref[slice_coverage*rot_n:slice_coverage*(rot_n+1),...]

            if motion_status[rot_n] == True:
                recon[slice_coverage * (rot_n) : slice_coverage * (rot_n + 1),...] = recon_previous[slice_coverage*rot_n:slice_coverage*(rot_n+1),...]

        recon_nb_image = np.rollaxis(recon,0,3)  
        print('under 1mm, recon shape: ', recon_nb_image.shape)
        print('max and min value: ', np.max(recon_nb_image), np.min(recon_nb_image))
        nb.save(nb.Nifti1Image(recon_nb_image,img_affine), os.path.join(random_folder,'image_data','recon.nii.gz'))

        # resample recon
        recon_1mm = nb.load(os.path.join(random_folder,'image_data','recon.nii.gz'))
        new_dim = [1,1,2.5]
        recon_resample = ff.resample_nifti(recon_1mm, order=3,  mode = 'nearest',  cval = np.min(recon_1mm.get_fdata()), in_plane_resolution_mm=new_dim[0], slice_thickness_mm=new_dim[-1])
        recon_resample = nb.Nifti1Image(recon_resample.get_fdata(), affine=recon_resample.affine, header=recon_resample.header)
        print('after 1mm, recon shape: ', recon_resample.get_fdata().shape)
        
        nb.save(recon_resample, os.path.join(random_folder,'image_data','recon_resample.nii.gz'))

        # get some statics
        motion = nb.load(os.path.join(random_folder, 'image_data','recon_resample.nii.gz')).get_fdata()
        static = nb.load(os.path.join(main_folder, 'simulations_202404','simulated_all_motion_v1',patient_id, patient_subid, 'static','image_data','recon_resample.nii.gz')).get_fdata()
        slice_to_compare = []
        for s in range(0, static.shape[2]):
            diff = np.mean(abs(motion[:,:,s] - static[:,:,s]))
            if diff >= 30:
                slice_to_compare.append(s)
        print('slice to compare: ', slice_to_compare)
        mae, mse, rmse, r_rmse, ssim = ff.compare(motion[:,:,slice_to_compare], static[:,:,slice_to_compare], cutoff_low = -100)
        print('mae: ', mae, ' rmse: ', rmse, ' ssim: ', ssim)

        
        
#         # # monitor the process
#         # max_value = np.max(recon_nb_image)
#         # min_value = np.min(recon_nb_image)
#         # print('max: ', max_value, 'min: ', min_value)
    
#         # # check_list.append([patient_id, patient_subid,random_i, max_value, min_value, with_double_skull])
#         # # df = pd.DataFrame(check_list, columns = ['patient_id', 'patient_subid','random' ,'max', 'min', 'with_double_skull'])
#         # # df.to_excel(os.path.join(main_save_folder,'check_list_static.xlsx'))

    
