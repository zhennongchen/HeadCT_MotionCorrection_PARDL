# PAR_corrected + iterative recon


#!/usr/bin/env python
import HeadCT_motion_correction_PAR.Data_processing as dp
import HeadCT_motion_correction_PAR.functions_collection as ff
from HeadCT_motion_correction_PAR.Build_lists import Build_list
import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.motion_simulator.transformation as transform
import HeadCT_motion_correction_PAR.motion_simulator.motion_simulation.ct_basic as basic
import ct_projector.projector.cupy as ct_projector

import importlib
import CTProjector.src.ct_projector.recon.cupy 
importlib.reload(CTProjector.src.ct_projector.recon.cupy)
import CTProjector.src.ct_projector.recon.cupy as ct_recon
import CTProjector.src.ct_projector.projector.cupy.fan_equiangular as ct_fan

import os
import numpy as np
import cupy as cp
import nibabel as nb
import pandas as pd


K = 12
geometry = 'fan'
total_view = 1400  ### 2340 views by default
gantry_rotation_time = 500 #unit ms, 500ms by default
view_increment =  56 #increment in gantry views
times = np.linspace(0, gantry_rotation_time, 5, endpoint=True)
spline_zeros = transform.interp_func(times, np.zeros([5,]))

cg = Defaults.Parameters()
trial_name = 'CNN_3D_motion_6degrees'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_3D_spline_6degrees_PAR_downsampled_slice0-50.xlsx')
save_folder = os.path.join(cg.predict_dir,trial_name,'images')

# build lists
print('Build List...')
b = Build_list.Build(data_sheet)
batches = [5]
batch_list, patient_id_list, patient_subid_list, random_name_list, start_slice_list, _, _, _, _, _, y_motion_param_predict, x_par_image_predict, y_image_predict  = b.__build__(batch_list = batches)
# n = np.arange(0,patient_id_list.shape[0], 3)
# m = ff.get_X_numbers_in_interval(n.shape[0],0,20, 100)
# n = n[m]
x_par_image_predict = x_par_image_predict[n]
y_image_predict = y_image_predict[n]
print(x_par_image_predict)


Quantitative_results = []
for i in range(0,x_par_image_predict.shape[0]):
    patient_subid = patient_subid_list[n[i]]
    patient_id = patient_id_list[n[i]]
    random_name = random_name_list[n[i]]

    save_sub = os.path.join(save_folder,patient_id, patient_subid, random_name)

    if random_name[0:3] == 'rea':
        print('real motion, skip now'); continue
    if os.path.isfile(os.path.join(save_sub, 'parameters/slice_5_to_20/pred_final.npy')) == 0:
        print('no pred parametes'); continue
    
    print(patient_id, patient_subid, random_name)

    # load gt image (static)
    gt_image = nb.load(os.path.join(cg.data_dir,'simulated_data_3D_spline',patient_id, patient_subid, 'static/image_data/recon_partial.nii.gz')).get_fdata()
    # load gt parameters
    gt_parameters = np.load(os.path.join(cg.data_dir,'simulated_data_3D_spline_6degrees',patient_id, patient_subid, random_name, 'motion_parameters.npy'),allow_pickle = True)
    spline_gt_tx = transform.interp_func(times, gt_parameters[0,:][0])
    spline_gt_ty = transform.interp_func(times, gt_parameters[1,:][0])
    spline_gt_tz = transform.interp_func(times, gt_parameters[2,:][0] / 2.5)
    spline_gt_rx = transform.interp_func(times, np.asarray(gt_parameters[3,:][0]) / 180 * np.pi)
    spline_gt_ry = transform.interp_func(times, np.asarray(gt_parameters[4,:][0]) / 180 * np.pi)
    spline_gt_rz = transform.interp_func(times, np.asarray(gt_parameters[5,:][0]) / 180 * np.pi)
    # load motion image
    motion_image = nb.load(os.path.join(cg.data_dir,'simulated_data_3D_spline_6degrees',patient_id, patient_subid, random_name,'image_data/recon_partial.nii.gz')).get_fdata()

    # quantitaive for motion
    mae_motion,_,rmse_motion,_, ssim_motion = ff.compare(motion_image[:,:,5:45], gt_image[:,:,5:45], cutoff_low = -10)
    print('motion image: ', mae_motion, rmse_motion, ssim_motion)

    # load parameters
    pred = np.load(os.path.join(save_sub,'parameters/slice_5_to_20/pred_final.npy'),allow_pickle = True)
    # 2D:
    # spline_x = transform.interp_func(times,np.concatenate([np.asarray([0]),pred[0,:]],axis = -1))
    # spline_y = transform.interp_func(times, np.concatenate([np.asarray([0]),pred[1,:]],axis = -1))
    # spline_r = transform.interp_func(times, np.concatenate([np.asarray([0]),pred[2,:]],axis = -1) / 180*np.pi)

    # 3D
    spline_pred_tx = transform.interp_func(times, np.concatenate([np.asarray([0]),pred[0,:]],axis = -1))
    spline_pred_ty = transform.interp_func(times, np.concatenate([np.asarray([0]),pred[1,:]],axis = -1))
    spline_pred_tz = transform.interp_func(times, np.concatenate([np.asarray([0]),ff.round_diff(pred[2,:], gt_parameters[2,:][0][1:] / 2.5, 0.5)],axis = -1))
    spline_pred_rx = transform.interp_func(times, np.concatenate([np.asarray([0]),ff.round_diff(pred[3,:], gt_parameters[3,:][0][1:], 0.5)],axis = -1) / 180 * np.pi)
    spline_pred_ry = transform.interp_func(times, np.concatenate([np.asarray([0]),ff.round_diff(pred[4,:], gt_parameters[4,:][0][1:], 0.5)],axis = -1) / 180 * np.pi)
    spline_pred_rz = transform.interp_func(times, np.concatenate([np.asarray([0]),pred[5,:]],axis = -1) / 180 * np.pi)

    # ###### step 1: re-make projection (so it has blank space at the beginning and the end)
    raw_img,spacing,img_affine = basic.basic_image_processing(os.path.join(cg.data_dir,'raw_data/nii-images/thin_slice',patient_id, patient_subid, 'img-nii-2.5/img.nii.gz'))
    raw_img = raw_img[0:60, :,:] # pick 0-60 slice
    # insert blank slices into beginning and end
    img = ff.insert_blank_slices(raw_img , insert_to_which_direction = 'x', begin_blank_slice_num = 5, end_blank_slice_num = 10)[np.newaxis, ...]
    print('img shape: ', img.shape)
    projector = basic.define_forward_projector(img,spacing,total_view); fbp_projector = basic.backprojector(img,spacing)
    sga = gt_parameters[6,:][0]; angles = ff.get_angles_zc(total_view, 360, sga)
    # FP:
    if 1==1:#os.path.isfile(os.path.join(save_sub,'pred_PAR_corrected.nii.gz')) == 0:
        projection = basic.fp_w_spline_motion_model(img, projector, angles, spline_gt_tx, spline_gt_ty, spline_gt_tz, spline_gt_rx, spline_gt_ry, spline_gt_rz, geometry, total_view = total_view, gantry_rotation_time = gantry_rotation_time, slice_num = None, increment = view_increment, order = 3)
    
    # ###### step 2: make PAR
    if 1==2:# os.path.isfile(os.path.join(save_sub,'pred_PAR_corrected.nii.gz')) == 1:
        # par_corrected_image = nb.load(os.path.join(save_sub,'pred_before_IR.nii.gz')).get_fdata()
        par_save_image = nb.load(os.path.join(save_sub,'pred_PAR_corrected.nii.gz')).get_fdata()

    else:
        sinogram_segments,_, num_angles_in_one_segment, segment_indexes = basic.divide_sinogram_new(projection, K , total_view)
        par = np.rollaxis(basic.make_PAR_new(sinogram_segments, segment_indexes, angles, img[0,...].shape, projector, fbp_projector, geometry),1,4)

        ###### step 3: apply inverse motion to PAR
        tt = np.linspace(int(gantry_rotation_time / (2*K + 1)), gantry_rotation_time, 2* K + 1, endpoint=True); par_c = np.copy(par)
        for j in range(0,par.shape[0]):
            I = par[j,...]
            _,_,_,transformation_matrix = transform.generate_transform_matrix([-spline_pred_tx(np.asarray([tt[j]])), -spline_pred_ty(np.asarray([tt[j]])) , -spline_pred_tz(np.asarray([tt[j]]))],[-spline_pred_rx(np.asarray([tt[j]])),-spline_pred_ry(np.asarray([tt[j]])), -spline_pred_rz(np.asarray([tt[j]]))],[1,1,1],I.shape, which_one_is_first='translation')
            transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, I.shape)
            par_c[j,...] = transform.apply_affine_transform(I, transformation_matrix,order = 3, cval = np.min(par))

        par_corrected_image = np.mean(par_c,axis=0)

        par_save_image = dp.cutoff_intensity(par_corrected_image[:,:,15:65], -1000)
         
        nb.save(nb.Nifti1Image(par_save_image, img_affine), os.path.join(save_sub,'pred_PAR_corrected.nii.gz'))
        # nb.save(nb.Nifti1Image(par_corrected_image, img_affine), os.path.join(save_sub,'pred_before_IR.nii.gz'))    

    # quantitaive for PAR_corrected
    mae_par,_,rmse_par,_, ssim_par = ff.compare(par_save_image[:,:,5:45], gt_image[:,:,5:45],cutoff_low = -10, extreme = 1000)
    print('par corrected: ', mae_par, rmse_par, ssim_par)


    ###### step 4: Iterative reconstruction
    # if os.path.isfile(os.path.join(save_sub,'IR/pred_IR_partial.nii.gz')):
    #     print('already done IR')
    #     continue
    #     # ir_corrected_image = nb.load(os.path.join(save_sub,'pred_IR.nii.gz')).get_fdata()
    #     # ir_save_image = nb.load(os.path.join(save_sub,'pred_IR_partial.nii.gz')).get_fdata()
    
    # else:
    #     # define projector
    #     projector_ir = projector
    #     curef = cp.array(img[0,...][:,np.newaxis,...], order='C')
    #     cuangles = cp.array(angles, order='C')
    #     projector_ir.set_projector(ct_fan.distance_driven_fp, angles=cuangles, branchless = False)
    #     projector_ir.set_backprojector(ct_fan.distance_driven_bp, angles=cuangles)
        
    #     # define initial image 
    #     if 1 == 2:#os.path.isfile(os.path.join(save_sub,'pred_IR.nii.gz')) == 1: # continue to iterate
    #         print('load previous IR, continue to iterate')
    #         initial = nb.load(os.path.join(save_sub,'pred_IR.nii.gz')).get_fdata()
    #         initial = (initial.astype(np.float32) + 1024) / 1000 * 0.019
    #         initial[initial < 0] = 0
    #         initial = np.rollaxis(initial,2,0)[:,np.newaxis,:,:]
    #         zero_init = False
            
    #     else:
    #         print('didnt do previous IR, start to iterate from PAR')
    #         zero_init = False
    #         initial = (par_corrected_image.astype(np.float32) + 1024) / 1000 * 0.019
    #         initial[initial < 0] = 0
    #         initial = np.rollaxis(initial,2,0)[:,np.newaxis,:,:]
        

    #     niter = 50
    #     nos = 5
    
    #     loss_base = 1000000000
    #     projector_norm = projector_ir.calc_projector_norm()
    #     cunorm_img = projector_ir.calc_norm_img() / projector_norm / projector_norm
        
    #     cuprj = cp.array(projection, cp.float32, order = 'C')

    #     if zero_init:
    #         curecon = cp.zeros(curef.shape, cp.float32)
    #         cunesterov = cp.zeros(curef.shape, cp.float32)
    #         print('zero start', curecon.shape)
    #     else:
    #         cuinitial = cp.array(initial ,order='C'); 
    #         curecon = cp.copy(cuinitial)
    #         cunesterov = cp.copy(cuinitial)

    #     ir_loss_record = []
    #     for iter in range(0,niter):
    #         for nn in range(0,nos):
    #             curecon, cunesterov, data_loss, _ = ct_recon.nesterov_acceleration_motion(
    #                 ct_recon.sqs_gaussian_one_step_motion,
    #                 img=curecon,
    #                 img_nesterov=cunesterov,
    #                 recon_kwargs={
    #                     'projector': projector_ir,
    #                     'prj': cuprj,
    #                     'norm_img': cunorm_img,
    #                     'projector_norm': projector_norm,
    #                     'beta': 0,
    #                     'spline_tx': spline_pred_tx,
    #                     'spline_ty': spline_zeros,
    #                     'spline_tz': spline_pred_tz,
    #                     'spline_rx': spline_pred_rx,
    #                     'spline_ry': spline_zeros, 
    #                     'spline_rz': spline_pred_rz, 
    #                     'sga': float(sga),
    #                     'total_view_num': total_view,
    #                     'increment': view_increment ,
    #                     'gantry_rotation_time': gantry_rotation_time
    #                     'return_loss': True,
    #                     'use_t_end': True,
    #                 }
    #             )
           

    #         ir_corrected_image = np.rollaxis(curecon.get()[:,0,:,:],0,3)/ 0.019 * 1000 - 1024
    #         ir_save_image = dp.cutoff_intensity(ir_corrected_image[:,:,15:65], -1000)

    #         # quantitaive for iterative recon
    #         mae_ir,_,rmse_ir,_, ssim_ir = ff.compare(ir_save_image[:,:,5:45],gt_image[:,:,5:45], cutoff_low = -10, extreme = 1000)
    #         print(iter, 'ir data_loss: ', data_loss, 'image_loss: ', mae_ir, rmse_ir, ssim_ir)

    #         if iter > 10:
    #             if (mae_ir >= loss_base):
    #                 break
                
    #         loss_base = mae_ir

    #         ir_loss_record.append([iter, data_loss, mae_ir, ssim_ir])
    #         df = pd.DataFrame(ir_loss_record, columns = ['step', 'data_loss', 'MAE', 'SSIM'])

    #         ir_folder = os.path.join(save_sub,'IR'); ff.make_folder([ir_folder])

    #         df.to_excel(os.path.join(ir_folder, 'IR_loss.xlsx'), index = False)

    #         nb.save(nb.Nifti1Image(ir_save_image ,img_affine), os.path.join(ir_folder,'pred_IR_partial.nii.gz'))
    #         nb.save(nb.Nifti1Image(ir_corrected_image ,img_affine), os.path.join(ir_folder,'pred_IR.nii.gz'))



    # Quantitative_results.append([batch_list[n[i]],patient_id, patient_subid, random_name, 
    #                 mae_motion, rmse_motion, ssim_motion,mae_par, rmse_par, ssim_par])

    # df = pd.DataFrame(Quantitative_results, columns = ['batch','Patient_ID', 'AccessionNumber', 'motion_name', 
    #             'mae_motion', 'rmse_motion', 'ssim_motion', 'mae_PAR', 'rmse_PAR', 'ssim_PAR' ])
    # df.to_excel(os.path.join(cg.predict_dir, trial_name,'comparison_images_test_PAR_corrected.xlsx'), index = False) 


        





   