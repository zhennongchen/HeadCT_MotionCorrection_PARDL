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
motion_type = '3D_spline_6degrees_HR'
motion_dim = 2#
amplitude_max = 5
displacement_max = 3

change_direction_limit = 2
CP_num = 5

geometry = 'fan'
total_view = 1400  ### 2340 views by default
gantry_rotation_time = 500 #unit ms, 500ms by default
view_increment = 28 # increment in gantry views


# define the patient list
data_folder = os.path.join(cg.data_dir,'raw_data/nii-images/thin_slice')
main_save_folder = os.path.join(cg.data_dir,'2D_motions/simulated_data_2D_spline')
par_save_folder = os.path.join(cg.data_dir, 'PAR_3D_spline_new_HR')
folder_name = 'random_'
batch_data_list = pd.read_excel(os.path.join(cg.data_dir, 'Patient_list/Patient_list_batch_PAR.xlsx'))


patient_list= ff.find_all_target_files(['*/*'],data_folder)
print(len(patient_list))

check_list = []
for p in patient_list:
    patient_subid = os.path.basename(p)
    patient_id = os.path.basename(os.path.dirname(p))

    if [patient_id, patient_subid] in motion_cases:
        print('already have motion. only recon original image'); L = [0]
        continue

    # find its batch
    batch = batch_data_list[(batch_data_list['PatientID'] == patient_id) & (batch_data_list['AccessionNumber'] == patient_subid)]['batch']
   
    print('patient: ',patient_id, patient_subid)
    save_folder = os.path.join(main_save_folder,patient_id,patient_subid)
    ff.make_folder([os.path.dirname(save_folder),save_folder])

    img_file = ff.find_all_target_files(['img-nii-0.625/img.nii.gz'],p)
    if len(img_file) != 1:
        ValueError('no raw data')
    
    img,spacing,img_affine = ct.basic_image_processing(img_file[0])
    # spacing = [0.625, 1.0, 1.0]
    print('nib image shape: ',img.shape, ' spacing: ',spacing)

    # define projectors
    img = img[np.newaxis, ...]
    projector = ct.define_forward_projector(img,spacing,total_view)
    fbp_projector = ct.backprojector(img,spacing)
    
    # very important - make sure that the arrays are saved in C order
    cp.cuda.Device(0).use()
    ct_projector.set_device(0)

    # do simulation
    L = np.arange(36,41)
    
    for random_i in L:
        t = np.linspace(0, gantry_rotation_time, CP_num, endpoint=True)
        # create folder
        random_folder = os.path.join(save_folder,folder_name +str(random_i))
        ff.make_folder([random_folder, os.path.join(random_folder,'image_data')])
        print('\n',random_i  , 'random')

        # check whether done:
        # if os.path.isfile(os.path.join(par_save_folder,patient_id, patient_subid, 'random_'+str(random_i),'all_slices/PARs_ds_crop_anneal.nii.gz')) == 1:
        if os.path.isfile(os.path.join(random_folder,'image_data/recon_partial.nii.gz')) == 1:
            print('done')
            continue

        else:
            # use previous motion:
            previous_motion_parameter_file = os.path.join(cg.data_dir, 'simulated_data_3D_spline_6degrees',patient_id, patient_subid, 'random_'+str(random_i), 'motion_parameters.npy')
            if 1==2:#os.path.isfile(previous_motion_parameter_file) == 1:
                print('use previous motion parameters')
                previous_motion_parameter = np.load(previous_motion_parameter_file, allow_pickle = True)
                amplitude_tx_mm = previous_motion_parameter[0,:][0]
                amplitude_ty_mm = previous_motion_parameter[1,:][0]
                amplitude_tz_mm = previous_motion_parameter[2,:][0]
                amplitude_rx_degree = previous_motion_parameter[3,:][0]
                amplitude_ry_degree = previous_motion_parameter[4,:][0]
                amplitude_rz_degree = previous_motion_parameter[5,:][0]
                sga = previous_motion_parameter[6,:][0]
            else:
                while True:
                    amplitude_tx_mm = transform.motion_control_point_generation(1, CP_num, amplitude_max, displacement_max, change_direction_limit, print_result = True)[:,0]
                    amplitude_ty_mm = transform.motion_control_point_generation(1, CP_num, amplitude_max, displacement_max, change_direction_limit, print_result = True)[:,0]
                    amplitude_tz_mm = transform.motion_control_point_generation(1, CP_num, 0,0, change_direction_limit, print_result = True)[:,0]
                    amplitude_rx_degree = transform.motion_control_point_generation(1, CP_num, 0,0, change_direction_limit, print_result = True)[:,0]
                    amplitude_ry_degree = transform.motion_control_point_generation(1, CP_num, 0,0, change_direction_limit, print_result = True)[:,0]
                    amplitude_rz_degree = transform.motion_control_point_generation(1, CP_num, amplitude_max, displacement_max, change_direction_limit, print_result = True)[:,0]
                    if np.max(abs(amplitude_rx_degree))+ np.max(abs(amplitude_ry_degree)) <= 7:
                        print('rx+ry: ', np.max(abs(amplitude_rx_degree))+ np.max(abs(amplitude_ry_degree)))
                        break

                    # mild
                    # amplitude_t_mm = transform.motion_control_point_generation(3, CP_num, amplitude_max, displacement_max, change_direction_limit, print_result = False)
                    # amplitude_tx_mm = amplitude_t_mm[:,0]; amplitude_ty_mm = amplitude_t_mm[:,1]; amplitude_tz_mm = amplitude_t_mm[:,2]
                    # amplitude_r_degree = transform.motion_control_point_generation(3, CP_num, amplitude_max, displacement_max, change_direction_limit, print_result = False)
                    # amplitude_rx_degree = amplitude_r_degree[:,0]; amplitude_ry_degree = amplitude_r_degree[:,1]; amplitude_rz_degree = amplitude_r_degree[:,2]
                  
                    # if np.max(abs(amplitude_rx_degree))+ np.max(abs(amplitude_ry_degree)) <= 5 and np.max(abs(amplitude_tz_mm)) <=2.5:
                    #     print('amplitude_tx_mm: ', amplitude_tx_mm, ' max: ', np.max(abs(amplitude_tx_mm)))
                    #     print('amplitude_ty_mm: ', amplitude_ty_mm, ' max: ', np.max(abs(amplitude_ty_mm)))
                    #     print('amplitude_tz_mm: ', amplitude_tz_mm, ' max: ', np.max(abs(amplitude_tz_mm)))
                    #     print('amplitude_rx_degree: ', amplitude_rx_degree, ' max: ', np.max(abs(amplitude_rx_degree)))
                    #     print('amplitude_ry_degree: ', amplitude_ry_degree, ' max: ', np.max(abs(amplitude_ry_degree)))
                    #     print('amplitude_rz_degree: ', amplitude_rz_degree, ' max: ', np.max(abs(amplitude_rz_degree)))
                    #     break
                    
                    
                sga = int(np.random.uniform(0,90))


            parameter_file = os.path.join(random_folder,'motion_parameters.txt')
            ff.txt_writer(parameter_file,[t.tolist(),amplitude_tx_mm, amplitude_ty_mm, amplitude_tz_mm, amplitude_rx_degree, amplitude_ry_degree, amplitude_rz_degree, [sga],[total_view],[gantry_rotation_time]],['time_points','translation_x_CP','translation_y_CP','translation_z_CP', 'rotation_x_CP', 'rotation_y_CP','rotation_z_CP','starting_gantry_angle', 'total_projection_view','gantry_rotation_time(ms)'])
            parameter_file = os.path.join(random_folder,'motion_parameters.npy')
            np.save(parameter_file, np.array([[amplitude_tx_mm],[amplitude_ty_mm],[amplitude_tz_mm],[amplitude_rx_degree],[amplitude_ry_degree],[amplitude_rz_degree],[sga], [t], [total_view], [gantry_rotation_time]], dtype=object))
            
            # prepare spline fit
            spline_tx = transform.interp_func(t, np.asarray([i/spacing[1] for i in amplitude_tx_mm]))
            spline_ty = transform.interp_func(t, np.asarray([i/spacing[2] for i in amplitude_ty_mm]))
            spline_tz = transform.interp_func(t, np.asarray([i/spacing[0] for i in amplitude_tz_mm]))
            spline_rx = transform.interp_func(t,np.asarray([i / 180 * np.pi for i in amplitude_rx_degree]))
            spline_ry = transform.interp_func(t,np.asarray([i / 180 * np.pi for i in amplitude_ry_degree]))
            spline_rz = transform.interp_func(t,np.asarray([i / 180 * np.pi for i in amplitude_rz_degree]))

            angles = ff.get_angles_zc(total_view, 360 ,sga)
    
            # generate forward projection
            projection = ct.fp_w_spline_motion_model(img, projector, angles, spline_tx, spline_ty, spline_tz, spline_rx, spline_ry, spline_rz, geometry, total_view = total_view, gantry_rotation_time = gantry_rotation_time, slice_num = None, increment = view_increment, order = 3)
            # save fp
            # projection_save_version = nb.Nifti1Image(projection[:,:,0,:], img_affine)
            # nb.save(projection_save_version, os.path.join(random_folder,'projection.nii.gz'))

            # # generate backprojection
            recon = ct.filtered_backporjection(projection,angles,projector,fbp_projector, geometry, back_to_original_value=True)
           
            # # save recon
            recon_nb_image = np.rollaxis(recon,0,3) 
            recon_nb_image = recon_nb_image[:,:,0:60]
            print(recon_nb_image.shape)
            # nb.save(nb.Nifti1Image(recon_nb_image,img_affine), os.path.join(random_folder,'image_data','recon.nii.gz'))
            nb.save(nb.Nifti1Image(recon_nb_image,img_affine), os.path.join(random_folder,'image_data','recon_partial.nii.gz'))

            # check the image max and min and the mae ssim to static
            max_value = np.max(recon_nb_image)
            min_value = np.min(recon_nb_image)
            # static = nb.load(os.path.join(cg.data_dir, 'simulated_data_3D_spline_6degrees_HR', patient_id, patient_subid, 'static', 'image_data/recon.nii.gz')).get_fdata()
            # mae_all, _, rmse_all, _, ssim_all = ff.compare(recon_nb_image[:,:,0 + 20 : recon_nb_image.shape[-1] - 20], static[:,:,0 + 20 : recon_nb_image.shape[-1] - 20], cutoff_low=-100)
            # print('mae_all: ', mae_all, ' rmse_all: ', rmse_all, ' ssim_all: ', ssim_all)
            # print('max: ', max_value, ' min: ', min_value)

            check_list.append([patient_id, patient_subid, max_value, min_value])
            df = pd.DataFrame(check_list, columns = ['patient_id', 'patient_subid', 'max', 'min'])
            df.to_excel(os.path.join(main_save_folder,'check_list31-35.xlsx'))


            # ## make PAR if needed
            # K = 12
            # save_folder2 = os.path.join(par_save_folder,patient_id, patient_subid, 'random_' + str(random_i),'all_slices')
            # ff.make_folder([os.path.dirname(os.path.dirname(os.path.dirname(save_folder2))), os.path.dirname(os.path.dirname(save_folder2)), os.path.dirname(save_folder2), save_folder2])
            # sinogram_segments, center_angle_index, num_angles_in_one_segment, segment_indexes = ct.divide_sinogram_new(projection, K , total_view)

            # PAR_collections = ct.make_PAR_new(sinogram_segments, segment_indexes, angles, img[0,...].shape, projector, fbp_projector, 'fan')
            # PAR_collections = np.rollaxis(PAR_collections,1,4)
            # # print(PAR_collections.shape)

            # PAR_collections_ds = block_reduce(PAR_collections, block_size=(1,2,2,1), func=np.mean)
            # print(PAR_collections_ds.shape)
            # # save_folder_sub = os.path.join(save_folder2); ff.make_folder([save_folder_sub])


            # crop_img = np.zeros([2*K + 1, 128,128 , PAR_collections_ds.shape[-1]])
            # for j in range(0,PAR_collections_ds.shape[0]):
            #     crop_img[j,...] = dp.crop_or_pad(PAR_collections_ds[j,...], [128,128,PAR_collections_ds.shape[-1] ], np.min(PAR_collections_ds[j,...]))
            # print('crop final image shape: ' ,crop_img.shape)

            # # anneal for faster training
            # new_par = np.zeros([25,128,128,75])
            # new_par[:,:,:, 0:25] = crop_img[:,:,:, 55:80]
            # new_par[:,:,:,25:50] = crop_img[:,:,:, 115:140]
            # new_par[:,:,:, 50:75] = crop_img[:,:,:, 175:200]
            # print('new par saved shape: ' ,new_par.shape)

            # nb.save(nb.Nifti1Image(new_par.astype(np.int32), img_affine), os.path.join(save_folder2, 'PARs_ds_crop_anneal.nii.gz'))
            # # nb.save(nb.Nifti1Image(crop_img, img_affine), os.path.join(save_folder2,'PARs_ds_crop.nii.gz'))
         

      


    

