import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nb
import SimpleITK as sitk
from scipy.ndimage import zoom


import HeadCT_motion_correction_PAR.functions_collection as ff
import HeadCT_motion_correction_PAR.Data_processing as dp
import HeadCT_motion_correction_PAR.motion_simulator.motion_simulation.ct_basic as basic
import HeadCT_motion_correction_PAR.motion_simulator.transformation as transform

import CTProjector.src.ct_projector.projector.numpy as ct_projector
import CTProjector.src.ct_projector.projector.numpy.fan_equiangluar as ct_fan
import CMR_HFpEF_Analysis.motion_correction.Bspline as Bspline
main_path = '/mnt/mount_zc_NAS/head_phantom_raw/processed'

study_list = ff.find_all_target_files(['study_3','study_5', 'study_6', 'study_8', 'study_10' ], '/mnt/mount_zc_NAS/motion_correction/data/phantom_data')

for study_folder in study_list:
    study = os.path.basename(study_folder)
    print(study)
    main_folder = study_folder

    if os.path.isfile(os.path.join(main_folder,'PAR_corrected_HRxy.nii.gz' )) == 1:
        print('already did'); continue
    

    motion_gt = np.load(os.path.join(main_folder, 'motion_each_point.npy'),allow_pickle = True)
    motion_pred = np.load(os.path.join('/mnt/mount_zc_NAS/motion_correction/predict/phantom_data/images', study, 'parameters/pred_final.npy'), allow_pickle = True)

    # get the spline function
    # gt motion:
    spline_tx_gt = transform.interp_func(np.linspace(0,100,25), np.asarray(motion_gt[0,:]))
    spline_tz_gt = transform.interp_func(np.linspace(0,100,25), np.asarray(motion_gt[2,:]))
    spline_rx_gt = transform.interp_func(np.linspace(0,100,25), np.asarray(motion_gt[3,:]))
    spline_rz_gt = transform.interp_func(np.linspace(0,100,25), np.asarray(motion_gt[5,:]))

    # pred motion
    spline_tx_pred = transform.interp_func(np.linspace(0,100,5), np.concatenate([np.asarray([0]),motion_pred[0,:]],axis = -1))
    spline_tz_pred = transform.interp_func(np.linspace(0,100,5), np.concatenate([np.asarray([0]),motion_pred[1,:]],axis = -1))
    spline_rx_pred = transform.interp_func(np.linspace(0,100,5), np.concatenate([np.asarray([0]),motion_pred[2,:]],axis = -1))
    spline_rz_pred = transform.interp_func(np.linspace(0,100,5), np.concatenate([np.asarray([0]),motion_pred[3,:]],axis = -1))

    prjs_new = nb.load(os.path.join(main_folder,'projection_HR.nii.gz')).get_fdata()
    angles = ff.get_angles_zc(1440, 360, 0)
    projector = ct_projector.ct_projector()
    projector.from_file('./projector_fan_scanner.cfg')
    projector.nv = 1
    projector.nz = 1
    projector.nx = 512
    projector.ny = 512
    projector.dx = 0.601
    projector.dy = 0.601

    # make PAR-corrected:
    img = np.zeros([432,512,512])
    for jj in range(0,3):
        if jj == 0:
            slice1 = 0; slice2 = 180
        if jj == 1:
            slice1 = 180; slice2 = 380
        if jj == 2:
            slice1 = 380; slice2 = 432

        projection = prjs_new[slice1:slice2,:,np.newaxis,:] #0-180, 180-380, 380-432
        sinogram_segments, center_angle_index, num_angles_in_one_segment, segment_indexes = basic.divide_sinogram_new(projection, 7 ,1440 , fill_out = False)

        PAR_collections = np.zeros([sinogram_segments.shape[0], slice2 - slice1, 512,512])

        for i in range(0,sinogram_segments.shape[0]):
            s = segment_indexes[i]
            segment = sinogram_segments[i,...]
            
            angles_partial = angles[s[0]:s[1]]

            # backprojection
            recon = basic.filtered_backporjection(np.copy(segment, 'C'),angles_partial,projector,projector,'fan', back_to_original_value = False)
            recon = recon / 0.0193 * 1000 - 1000
            
            PAR_collections[i,:,:,:] = recon.astype(np.int32)

        final_par_c = np.zeros_like(PAR_collections)
        t = np.linspace(0,100,15)

        for i in range(0, PAR_collections.shape[0]):
            # motion:
            # tx = np.round(spline_tx_pred([t[i]]) / 0.601) * 0.601
            tz = np.round(spline_tz_pred([t[i]]) / 0.707)  * 0.707
            rz = np.round(spline_rz_pred([t[i]]) / 0.25)  * 0.25

            I = PAR_collections[i,...]
            _,_,_,transformation_matrix = transform.generate_transform_matrix([-tz / 0.707  , 0  / 0.601, 0 ],[-rz / 180 * np.pi, 0,0],[1,1,1],I.shape, which_one_is_first='translation')
            transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, I.shape)
            final_par_c[i,...] = transform.apply_affine_transform(I, transformation_matrix,order = 3, cval = np.min(I))

            print('done transformation: ',i, tz, rz)

        par_corrected_image = np.mean(final_par_c,axis=0)
        img[slice1: slice2,...] = par_corrected_image

        img_average = np.zeros([img.shape[0] // 4, img.shape[1], img.shape[2]])
        for i in range(0,img.shape[0] // 4):
            img_average[i,...] = np.mean(img[(4 * i) : (4 * i + 4),...], axis = 0)

        img_average = img_average.astype(np.int16)

        sitk_recon = sitk.GetImageFromArray(img_average)
        sitk_recon.SetSpacing([0.6016,0.6016, 2.828])
        sitk.WriteImage(sitk_recon, os.path.join(main_folder, 'PAR_corrected_HRxy.nii.gz')) 

