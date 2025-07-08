#!/usr/bin/env python
# since we know the ground truth motion, we can make the motion-corrected image 
# by forward-projecting the objecgt with delta motion 
# delta motion = ground_truth_motion - predicted_motion

import HeadCT_motion_correction_PAR.Data_processing as dp
import HeadCT_motion_correction_PAR.functions_collection as ff
from HeadCT_motion_correction_PAR.Build_lists import Build_list
import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.motion_simulator.transformation as transform
import HeadCT_motion_correction_PAR.motion_simulator.motion_simulation.ct_basic as ct
import ct_projector.projector.cupy as ct_projector
import ct_projector.projector.cupy.fan_equiangular as ct_fan
import ct_projector.projector.numpy as numpy_projector
import ct_projector.projector.numpy.fan_equiangluar as numpy_fan
import ct_projector.projector.cupy.parallel as ct_para
import ct_projector.projector.numpy.parallel as numpy_para

import os
import numpy as np
import nibabel as nb
import pandas as pd
import cupy as cp

geometry = 'fan'
total_view = 1400  ### 2340 views by default
gantry_rotation_time = 500 #unit ms, 500ms by default
view_increment = 28 # increment in gantry views
times = np.linspace(0, gantry_rotation_time, 5, endpoint=True)

cg = Defaults.Parameters()
trial_name = 'CNN_3D_motion_6degrees'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_3D_spline_6degrees_PAR_downsampled_slice0-50.xlsx')

save_folder = os.path.join(cg.predict_dir,trial_name,'images')

# build lists
print('Build List...')
b = Build_list.Build(data_sheet)
batches = [5]
batch_list, patient_id_list, patient_subid_list, random_name_list, start_slice_list, _, _, _, _, _, y_motion_param_predict, x_par_image_predict, _ = b.__build__(batch_list = batches)
n = np.arange(0,1,1)
x_par_image_predict = x_par_image_predict[n]
y_motion_param_predict = y_motion_param_predict[n]

results = []
for i in range(0,x_par_image_predict.shape[0]):
    patient_subid = patient_subid_list[n[i]]
    patient_id = patient_id_list[n[i]]
    random_name = random_name_list[n[i]]
    batch = batch_list[n[i]]

    save_sub = os.path.join(save_folder,patient_id, patient_subid, random_name)

    print(patient_id, patient_subid, random_name)

    if random_name[0:3] == 'rea':
        print('real motion, skip now'); continue
    if os.path.isfile(os.path.join(save_sub, 'parameters/slice_5_to_20/pred_final.npy')) == 0:
        print('No predicted parameters!')
        continue

    if  1== 2:#os.path.isfile(os.path.join(save_sub,'pred.nii.gz')) == 1:
        print('already done, just load')
        pred_img = nb.load(os.path.join(save_sub,'pred.nii.gz')).get_fdata()

    else:
        ##### load gt parameters
        file_name = os.path.join(cg.data_dir,'simulated_data_3D_spline_6degrees',patient_id, patient_subid, random_name, 'motion_parameters.npy')
        gt = np.load(file_name, allow_pickle = True)
        # 3D:
        spline_gt_tx = transform.interp_func(times, gt[0,:][0])
        spline_gt_ty = transform.interp_func(times, gt[1,:][0])
        spline_gt_tz = transform.interp_func(times, gt[2,:][0] / 2.5)
        spline_gt_rx = transform.interp_func(times, np.asarray(gt[3,:][0]) / 180 * np.pi)
        spline_gt_ry = transform.interp_func(times, np.asarray(gt[4,:][0]) / 180 * np.pi)
        spline_gt_rz = transform.interp_func(times, np.asarray(gt[5,:][0]) / 180 * np.pi)

        # 2D:
        # spline_gt_tx = transform.interp_func(times, np.asarray(gt[0,:][0]))
        # spline_gt_ty = transform.interp_func(times, np.asarray(gt[2,:][0]))
        # spline_gt_theta = transform.interp_func(times, np.asarray(gt[5,:][0]) / 180 * np.pi)
    
        ##### load predicted parameters
        pred = np.load(os.path.join(save_sub,'parameters/slice_5_to_20/pred_final.npy'),allow_pickle = True)
        # 3D
        spline_pred_tx = transform.interp_func(times, np.concatenate([np.asarray([0]),pred[0,:]],axis = -1))
        spline_pred_ty = transform.interp_func(times, np.concatenate([np.asarray([0]),pred[1,:]],axis = -1))
        spline_pred_tz = transform.interp_func(times, np.concatenate([np.asarray([0]),ff.round_diff(pred[2,:], gt[2,:][0][1:]/2.5, 0.5)],axis = -1))
        spline_pred_rx = transform.interp_func(times, np.concatenate([np.asarray([0]),ff.round_diff(pred[3,:], gt[3,:][0][1:], 0.5)],axis = -1) / 180 * np.pi)
        spline_pred_ry = transform.interp_func(times, np.concatenate([np.asarray([0]),ff.round_diff(pred[4,:], gt[4,:][0][1:], 0.5)],axis = -1) / 180 * np.pi)
        spline_pred_rz = transform.interp_func(times, np.concatenate([np.asarray([0]),pred[5,:]],axis = -1) / 180 * np.pi)
        
        # 2D:
        # spline_pred_tx = transform.interp_func(times, np.concatenate([np.asarray([0]),pred[0,:]],axis = -1))
        # spline_pred_ty = transform.interp_func(times, np.concatenate([np.asarray([0]),pred[1,:]],axis = -1))
        # spline_pred_theta = transform.interp_func(times, np.concatenate([np.asarray([0]),pred[2,:]],axis = -1) / 180 * np.pi)

        # make forward projection using delta motion
        sga = gt[6,:][0]
        # sga = 0

        angles = ff.get_angles_zc(total_view, 360 ,sga)
        img,spacing,img_affine = ct.basic_image_processing(os.path.join(cg.data_dir,'raw_data/nii-images/thin_slice',patient_id, patient_subid, 'img-nii-2.5/img.nii.gz'))
        img = img[np.newaxis, ...]
        projector = ct.define_forward_projector(img,spacing,total_view)
        fbp_projector = ct.backprojector(img,spacing)
        
        # very important - make sure that the arrays are saved in C order
        cp.cuda.Device(0).use()
        ct_projector.set_device(0)
        
        # 3D:
        projection = ct.fp_w_delta_motion_model(img, projector, angles,  spline_gt_tx, spline_gt_ty, spline_gt_tz, spline_gt_rx, spline_gt_ry, spline_gt_rz,
                                                spline_pred_tx, spline_pred_ty, spline_pred_tz, spline_pred_rx, spline_pred_ry, spline_pred_rz,
                                                geometry, total_view = total_view, gantry_rotation_time = gantry_rotation_time, slice_num = None, increment = view_increment, order = 3)

        # projection = ct.fp_w_delta_motion_model(img, projector, angles,  spline_gt_tx, spline_gt_ty, spline_zeros, spline_zeros, spline_zeros, spline_gt_theta,
        #                                         spline_pred_tx, spline_pred_ty, spline_zeros, spline_zeros, spline_zeros, spline_pred_theta,
        #                                         geometry, total_view = total_view, gantry_rotation_time = gantry_rotation_time, slice_num = None, increment = view_increment, order = 3)
            
        pred_img = ct.filtered_backporjection(projection,angles,projector,fbp_projector, geometry, back_to_original_value=True)
        pred_img = np.rollaxis(pred_img,0,3)[:,:,10:60]
        # pred_img = np.rollaxis(pred_img,0,3)[:,:,0:60]

        # if ff.find_timeframe(random_name,0,'_') <= 10:
        nb.save(nb.Nifti1Image(pred_img,img_affine), os.path.join(save_sub,'pred_PAR_corrected.nii.gz'))

    ##### quantitative 
    # load gt image (static)
    file_name = os.path.join(cg.data_dir,'simulated_data_3D_spline',patient_id, patient_subid, 'static/image_data/recon_partial.nii.gz')
    gt_img = nb.load(file_name).get_fdata()

    # load motion image
    file_name = os.path.join(cg.data_dir,'simulated_data_3D_spline_6degrees',patient_id, patient_subid, random_name, 'image_data/recon_partial.nii.gz')
    motion_img = nb.load(file_name).get_fdata()

    mae_motion,_,rmse_motion,_, ssim_motion = ff.compare(motion_img[:,:,5:45], gt_img[:,:,5:45] ,cutoff_low = -10)
    mae_pred,_,rmse_pred,_, ssim_pred = ff.compare(pred_img[:,:,5:45], gt_img[:,:,5:45] ,cutoff_low = -10,extreme = 1000)

    print(mae_motion, rmse_motion, ssim_motion)
    print(mae_pred, rmse_pred, ssim_pred)
    # print(mae_motion, rmse_motion, ssim_motion, mae_pred, rmse_pred,  ssim_pred)

    # results.append([batch,patient_id, patient_subid, random_name, 
    #                 mae_motion, rmse_motion, ssim_motion, mae_pred, rmse_pred,  ssim_pred,])


    # df = pd.DataFrame(results, columns = ['batch','Patient_ID', 'AccessionNumber', 'motion_name', 
    #                 'mae_motion', 'rmse_motion', 'ssim_motion', 'mae_pred', 'rmse_pred', 'ssim_pred'])
    # df.to_excel(os.path.join(cg.predict_dir, trial_name,'comparison_images_test_FP.xlsx'), index = False)       