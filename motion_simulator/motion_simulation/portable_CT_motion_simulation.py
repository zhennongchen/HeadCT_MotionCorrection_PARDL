#!/usr/bin/env python

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


cg = Defaults.Parameters()
motion_cases = [['MO101701M000006', 'MO001A000073'],['MO101701M000006', 'MO001A000073_2'],
                ['MO101701M000006', 'MO001A000128']  ,['MO101701M000010', 'MO001A000123'],
                ['MO101701M000021','MO001A000048'], ['MO101701M000014','MO001A000047'],['MO101701M000014','MO001A000112']
                ]

# define motion range:
motion_type = 'simulated_data_portable_CT_new'
amplitude_max_severe = 8
displacement_max_severe = 5
amplitude_max_mild = 5
displacement_max_mild = 3
# offset_max = 5

motion_freq = 1.1 # 0.3 or 0.35
severe_freq = 0.25 # 0.6 or 0.65
double_skull_freq = 0.1 # 0.3 or 0.35

change_direction_limit = 2
CP_num = 5

geometry = 'fan'
total_view = 1400  ### 2340 views by default
gantry_rotation_time = 500 #unit ms, 500ms by default
view_increment = 56 # increment in gantry views

# define the patient list
data_folder = os.path.join(cg.data_dir,'raw_data/nii-images/thin_slice')
main_save_folder = os.path.join(cg.data_dir,motion_type)
folder_name = 'random_'

patient_list= ff.find_all_target_files(['*/*'],data_folder)
print('patient list len: ', len(patient_list))

check_list = []
for p in patient_list:
    patient_subid = os.path.basename(p)
    patient_id = os.path.basename(os.path.dirname(p))

    if [patient_id, patient_subid] in motion_cases:
        print('already have motion. only recon original image'); L = [0]
        continue
   
    print('patient: ',patient_id, patient_subid)
    save_folder = os.path.join(main_save_folder,patient_id,patient_subid)
    ff.make_folder([os.path.dirname(save_folder),save_folder])

    img_file = ff.find_all_target_files(['img-nii-0.625/img.nii.gz'],p)
    if len(img_file) != 1:
        ValueError('no raw data')
    
    img,spacing,img_affine = ct.basic_image_processing(img_file[0])
    print('nib image shape: ',img.shape, ' spacing: ',spacing)

    # define projectors
    img = img[np.newaxis, ...]
    projector = ct.define_forward_projector(img,spacing,total_view)
    fbp_projector = ct.backprojector(img,spacing)
    
    # very important - make sure that the arrays are saved in C order
    cp.cuda.Device(0).use()
    ct_projector.set_device(0)

    # load the static image as reference
    if os.path.isfile(os.path.join(cg.data_dir, 'simulated_data_portable_CT', patient_id, patient_subid, 'static','image_data','recon.nii.gz')) == 0:
        static_ref = None
    else:
        static_ref= nb.load(os.path.join(cg.data_dir, 'simulated_data_portable_CT', patient_id, patient_subid, 'static','image_data','recon.nii.gz')).get_fdata()
        static_ref = np.rollaxis(static_ref,2,0)
        print('static ref shape: ', static_ref.shape)
    print('static_ref is None? ', static_ref is None)

    # do simulation
    L = np.arange(16,17)

    for random_i in L:
        t = np.linspace(0, gantry_rotation_time, CP_num, endpoint=True)
        # create folder
        if random_i == 0:
            random_folder = os.path.join(save_folder,'static')
        else:
            random_folder = os.path.join(save_folder,folder_name +str(random_i))
        ff.make_folder([random_folder, os.path.join(random_folder,'image_data')])
        print('\n',random_i  , 'random')

        if os.path.isfile(os.path.join(random_folder,'image_data','recon.nii.gz')) == 1:
            print('already done this motion')
            continue

        # gantry coverage = 1cm, which means it rotates once for every 1cm interval in the z-axis
        # first set the starting gantry angle (SGA)
        sga = int(np.random.uniform(0,90))
        # second, find out how many rotations we need
        rotation_num = int(spacing[0] * img.shape[1] // 10)
        slice_coverage = int(10 // 0.625)  # slice coverage = 1cm

        # start to generate the motion for each rotation
        # set whether static or motion for each rotation
        while True:
            motion_status = [np.random.uniform(0,1) <motion_freq for k in range(rotation_num-3)]
            if motion_freq <=0:
                break
            if np.sum(motion_status) > 0:
                break
        motion_status = [False, False] + motion_status + [False] # add two static rotations at the beginning and end
        print('motion_status: ', motion_status)
        
        amplitude_collect = np.zeros([rotation_num, CP_num, 6])
        recon = np.zeros([slice_coverage * rotation_num,img.shape[2],img.shape[3]])
        # projection = np.zeros([slice_coverage * rotation_num, total_view, 1 , projector.nu])

        for rot_n in range(0,rotation_num):
            # print('rotation: ', rot_n)
            # set the partial image:
            slice_start = rot_n * slice_coverage
            slice_end = (rot_n + 1) * slice_coverage
            slice_prior = max(0, slice_start - slice_coverage * 2)
            slice_post = min(img.shape[1], slice_end + slice_coverage * 2)
            img_partial = img[:,slice_prior:slice_post,:]

            # set the motion 
            # first: determine whether in this rotation we have motion or static, and whether it's severe or mild
            if motion_status[rot_n]:
                amplitude_max_tem_r = amplitude_max_mild # rotation always max =  5 degree
                displacement_max_tem_r = displacement_max_mild

                if np.random.uniform(0,1) < severe_freq: # severe motion
                    severe_motion = True
                    amplitude_max_tem_t = amplitude_max_severe
                    displacement_max_tem_t = displacement_max_severe
                else: # mild motion
                    severe_motion = False
                    amplitude_max_tem_t = amplitude_max_mild 
                    displacement_max_tem_t = displacement_max_mild
            else:
                severe_motion = False
                amplitude_max_tem_t, displacement_max_tem_t, amplitude_max_tem_r, displacement_max_tem_r = 0, 0, 0, 0

            # second: determine whether the initial pose has been changed compared to last rotation
            offset_values = [0] * 6
            
            # third: generate the motion
            # translation
            amplitude_txty_mm = transform.motion_control_point_generation(2, CP_num, amplitude_max = amplitude_max_tem_t, displacement_max = displacement_max_tem_t, change_direction_limit = change_direction_limit, offset_value = offset_values[0:2], print_result =False)
            amplitude_tx_mm = amplitude_txty_mm[:,0]
            amplitude_ty_mm = amplitude_txty_mm[:,1]
            if severe_motion:
                amplitude_tz_mm = transform.motion_control_point_generation(1, CP_num, amplitude_max = 3, displacement_max = 1.5, change_direction_limit = change_direction_limit, offset_value = offset_values[2], print_result =False)[:,0]
            elif not severe_motion and motion_status[rot_n]:
                amplitude_tz_mm = transform.motion_control_point_generation(1, CP_num, amplitude_max = 1.5, displacement_max = 1, change_direction_limit = change_direction_limit, offset_value = offset_values[2], print_result =False)[:,0]
            else:
                amplitude_tz_mm = transform.motion_control_point_generation(1, CP_num, amplitude_max = 0, displacement_max = 0, change_direction_limit = change_direction_limit, offset_value = offset_values[2], print_result =False)[:,0]
            
            # rotations
            while True:
                if severe_motion:
                    amplitude_rxryrz_degree = transform.motion_control_point_generation(3, CP_num, amplitude_max = 7, displacement_max = 5, change_direction_limit = change_direction_limit, offset_value = offset_values[3:6], print_result =False)
                else:
                    amplitude_rxryrz_degree = transform.motion_control_point_generation(3, CP_num, amplitude_max = amplitude_max_tem_r, displacement_max = displacement_max_tem_r, change_direction_limit = change_direction_limit, offset_value = offset_values[3:6], print_result =False)
                amplitude_rx_degree = amplitude_rxryrz_degree[:,0]
                amplitude_ry_degree = amplitude_rxryrz_degree[:,1]
                amplitude_rz_degree = amplitude_rxryrz_degree[:,2]
                if np.max(abs(amplitude_rx_degree - amplitude_rx_degree[0]))+ np.max(abs(amplitude_ry_degree - amplitude_ry_degree[0])) <= 6:
                    break

            # let's also consider the double skull artifacts, espeically in the occipital bone
            if np.random.uniform(0,1) < double_skull_freq and motion_status[rot_n]:
                print('yes we have double skull')
                amplitude_ty_mm = transform.motion_control_point_generation(1, CP_num, amplitude_max = 12, displacement_max = 8, change_direction_limit = change_direction_limit, offset_value = offset_values[1], print_result =False)[:,0]
                amplitude_tx_mm = transform.motion_control_point_generation(1, CP_num, amplitude_max = 5, displacement_max = 3, change_direction_limit = change_direction_limit, offset_value = offset_values[0], print_result =False)[:,0]
                amplitude_tz_mm = transform.motion_control_point_generation(1, CP_num, amplitude_max = 1, displacement_max = 0.5, change_direction_limit = change_direction_limit, offset_value = offset_values[2], print_result =False)[:,0]

            # print('amplitude_tx_mm: ', amplitude_tx_mm)
            # print('amplitude_ty_mm: ', amplitude_ty_mm)
            # print('amplitude_tz_mm: ', amplitude_tz_mm)
            # print('amplitude_rx_degree: ', amplitude_rx_degree)
            # print('amplitude_ry_degree: ', amplitude_ry_degree)
            # print('amplitude_rz_degree: ', amplitude_rz_degree)
     
            # save the motion parameters
            parameter_file = os.path.join(random_folder,'motion_parameters.txt')
            if rot_n == 0:
                ff.txt_writer(parameter_file, True, [t.tolist(),amplitude_tx_mm, amplitude_ty_mm, amplitude_tz_mm, amplitude_rx_degree, amplitude_ry_degree, amplitude_rz_degree, [sga],[total_view],[gantry_rotation_time]],['time_points','translation_x_CP','translation_y_CP','translation_z_CP', 'rotation_x_CP', 'rotation_y_CP','rotation_z_CP','starting_gantry_angle', 'total_projection_view','gantry_rotation_time(ms)'])
            else:
                ff.txt_writer(parameter_file, False, [t.tolist(),amplitude_tx_mm, amplitude_ty_mm, amplitude_tz_mm, amplitude_rx_degree, amplitude_ry_degree, amplitude_rz_degree, [sga],[total_view],[gantry_rotation_time]],['time_points','translation_x_CP','translation_y_CP','translation_z_CP', 'rotation_x_CP', 'rotation_y_CP','rotation_z_CP','starting_gantry_angle', 'total_projection_view','gantry_rotation_time(ms)'])
            
            collect = np.stack([amplitude_tx_mm, amplitude_ty_mm, amplitude_tz_mm, amplitude_rx_degree, amplitude_ry_degree, amplitude_rz_degree], axis = 1)
            amplitude_collect[rot_n,...] = collect
            
            # prepare spline fit
            spline_tx = transform.interp_func(t, np.asarray([i/spacing[1] for i in amplitude_tx_mm]))
            spline_ty = transform.interp_func(t, np.asarray([i/spacing[2] for i in amplitude_ty_mm]))
            spline_tz = transform.interp_func(t, np.asarray([i/spacing[0] for i in amplitude_tz_mm]))
            spline_rx = transform.interp_func(t,np.asarray([i / 180 * np.pi for i in amplitude_rx_degree]))
            spline_ry = transform.interp_func(t,np.asarray([i / 180 * np.pi for i in amplitude_ry_degree]))
            spline_rz = transform.interp_func(t,np.asarray([i / 180 * np.pi for i in amplitude_rz_degree]))

            angles = ff.get_angles_zc(total_view, 360 ,sga)

            # if static and has the static image ref, then no need to do the projection
            if motion_status[rot_n] == False:
                if static_ref is not None:
                    recon[slice_coverage * (rot_n) : slice_coverage * (rot_n + 1),...] = static_ref[slice_coverage*rot_n:slice_coverage*(rot_n+1),...]
                    continue

            # generate forward projection
            projection_partial = ct.fp_w_spline_motion_model(img_partial, projector, angles, spline_tx, spline_ty, spline_tz, spline_rx, spline_ry, spline_rz, geometry, total_view = total_view, gantry_rotation_time = gantry_rotation_time, slice_num = None, increment = view_increment, order = 3)
            projection_partial = projection_partial[slice_start - slice_prior : slice_end - slice_prior,...]
            
            # generate backprojection
            recon_partial = ct.filtered_backporjection(projection_partial,angles,projector,fbp_projector, geometry, back_to_original_value=True)
            recon[slice_coverage * (rot_n) : slice_coverage * (rot_n + 1),...] = recon_partial
            
            # save fp
            # projection_save_version = nb.Nifti1Image(projection[:,:,0,:], img_affine)
            # nb.save(projection_save_version, os.path.join(random_folder,'projection.nii.gz'))

        parameter_file = os.path.join(random_folder,'motion_parameters.npy')
        np.save(parameter_file, np.array([[amplitude_collect],[sga], [t], [total_view], [gantry_rotation_time]], dtype=object))

        # generate backprojection
        # recon = ct.filtered_backporjection(projection,angles,projector,fbp_projector, geometry, back_to_original_value=True)
           
        # save recon
        recon_nb_image = np.rollaxis(recon,0,3)  
        print(recon_nb_image.shape)
        nb.save(nb.Nifti1Image(recon_nb_image,img_affine), os.path.join(random_folder,'image_data','recon.nii.gz'))
        
        # # monitor the process
        # max_value = np.max(recon_nb_image)
        # min_value = np.min(recon_nb_image)
        # print('max: ', max_value, 'min: ', min_value)
    
        # # check_list.append([patient_id, patient_subid,random_i, max_value, min_value, with_double_skull])
        # # df = pd.DataFrame(check_list, columns = ['patient_id', 'patient_subid','random' ,'max', 'min', 'with_double_skull'])
        # # df.to_excel(os.path.join(main_save_folder,'check_list_static.xlsx'))
