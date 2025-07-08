import importlib
import CTProjector.src.ct_projector.recon.cupy 
importlib.reload(CTProjector.src.ct_projector.recon.cupy)

import numpy as np
import cupy as cp
import os
import pandas as pd
import nibabel as nb
import SimpleITK as sitk


import HeadCT_motion_correction_PAR.motion_simulator.motion_simulation.ct_basic as basic
import HeadCT_motion_correction_PAR.functions_collection as ff
import HeadCT_motion_correction_PAR.motion_simulator.transformation as transform

import CTProjector.src.ct_projector.projector.cupy as ct_projector
import CTProjector.src.ct_projector.recon.cupy as ct_recon
import CTProjector.src.ct_projector.projector.cupy.fan_equiangular as ct_fan

main_folder = '/mnt/mount_zc_NAS/motion_correction/data/phantom_data/study_3'

# load sinogram with motion
sinogram_file = nb.load(os.path.join(main_folder,'projection_HR_for_recon.nii.gz'))
sinogram = sinogram_file.get_fdata()
sinogram = sinogram[220:260,...]  # only pick 30 slices: 10-40th slice in 50-slice volume
sinogram = sinogram[:,:,np.newaxis,:]
print('sinogram size: ',sinogram.shape)
print(np.min(sinogram), np.max(sinogram))

# define motion
total_view_num = 1440
increment = 96
gantry_rotation_time = 500

t = np.linspace(gantry_rotation_time/ 15,gantry_rotation_time, 15, endpoint=True)
motion = np.load(os.path.join(main_folder, 'motion_each_point.npy'),allow_pickle = True)
spline_tx = transform.interp_func(np.linspace(0,gantry_rotation_time,25), np.asarray(motion[0,:])); spline_tx = transform.interp_func(t, spline_tx(np.linspace(0,gantry_rotation_time,15)))  # align the time frame
spline_ty = transform.interp_func(np.linspace(0,gantry_rotation_time,25), np.zeros([25]))
spline_tz = transform.interp_func(np.linspace(0,gantry_rotation_time,25), np.asarray(motion[2,:])); spline_tz = transform.interp_func(t, np.round(spline_tz(np.linspace(0,gantry_rotation_time,15)) / 0.707) * 0.707)
spline_rx = transform.interp_func(np.linspace(0,gantry_rotation_time,25), np.zeros([25]))
spline_ry = transform.interp_func(np.linspace(0,gantry_rotation_time,25), np.zeros([25]))
spline_rz = transform.interp_func(np.linspace(0,gantry_rotation_time,25), np.asarray(motion[5,:]) ); spline_rz = transform.interp_func(t, np.round(spline_rz(np.linspace(0,gantry_rotation_time,15)) / 0.25) * 0.25 / 180 * np.pi)

# define other parameters
sga = 0
angles = ff.get_angles_zc(1440, 360, 0)
projector = ct_projector.ct_projector()
projector.from_file('./projector_fan_scanner.cfg')
projector.nv = 1
projector.nz = 1
projector.nx = 308
projector.ny = 308
projector.dx = 1.
projector.dy = 1.

# load PAR_corrected LR
PAR_LR =  sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(main_folder,'PAR_corrected_gt.nii.gz' )))[10:20,...]
print(PAR_LR.shape)

PAR = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(main_folder,'PAR_corrected_gt_HR.nii.gz' )))[40:80,...]
print(PAR.shape)

# load static
static = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join('/mnt/mount_zc_NAS/head_phantom_raw/processed','study_4/scan4/processed_0', 'recon_cropped/recon_partial.nii.gz' )))[10:20,...]

mae_par,_,rmse_par,_, ssim_par = ff.compare(PAR_LR, static,cutoff_low = -10, extreme = 1000)
print('PAR results: ', mae_par, rmse_par, ssim_par)

# Iterative recon:
angles = ff.get_angles_zc(total_view_num, 360, sga)
cuangles = cp.array(angles, order='C')

projector.set_projector(ct_fan.distance_driven_fp, angles=cuangles, branchless = False)
projector.set_backprojector(ct_fan.distance_driven_bp, angles=cuangles)  # no FBP!!


# doing recon
PAR_corrected = (PAR.astype(np.float32) + 1000) / 1000 * 0.0193
PAR_corrected[PAR_corrected < 0] = 0
PAR_corrected = PAR_corrected[:,np.newaxis,...]
print(PAR_corrected.shape)

niter = 200
nos = 2
nesterov = 0.5
beta = 0.025
zero_init = False

projector_norm = projector.calc_projector_norm()
cunorm_img = projector.calc_norm_img() / projector_norm / projector_norm


cufbp = cp.array(PAR_corrected ,order='C')
cuprj2 = cp.array(sinogram, cp.float32, order = 'C')
cuangles = cp.array(angles, order='C')

Result = []

if zero_init:
    curecon = cp.zeros(cufbp.shape, cp.float32)
    cunesterov = cp.zeros(cufbp.shape, cp.float32)
else:    
    curecon = cp.copy(cufbp)
    cunesterov = cp.copy(curecon)

for i in range(0,niter):
    for n in range(0,nos):

        curecon, cunesterov, data_loss, _ = ct_recon.nesterov_acceleration_motion(
            ct_recon.sqs_gaussian_one_step_motion,
            img=curecon,
            img_nesterov=cunesterov,
            recon_kwargs={
                'projector': projector,
                'prj': cuprj2,
                'norm_img': cunorm_img,
                'projector_norm': projector_norm,
                'beta': beta,
                'spline_tx': spline_tx,
                'spline_ty': spline_ty,
                'spline_tz': spline_tz,
                'spline_rx': spline_rx,
                'spline_ry': spline_ry,
                'spline_rz': spline_rz,
                'sga': float(sga),
                'total_view_num': total_view_num,
                'increment': increment ,
                'gantry_rotation_time': gantry_rotation_time,
                'return_loss':  True,
                'use_t_end': True,
            }
        )
 
    recon_ir = curecon.get()[:,0,:,:]
    recon_ir = recon_ir / 0.0193 * 1000 - 1000

    recon_ir_LR = np.zeros([recon_ir.shape[0]//4, recon_ir.shape[1], recon_ir.shape[2]])
    for ss in range(0,recon_ir.shape[0] // 4):
        recon_ir_LR[ss,...] = np.mean(recon_ir[(4 * ss) : (4 * ss + 4),...], axis = 0)
    print('recon shape: ',recon_ir_LR.shape)

    recon_ir_LR_save = ff.insert_blank_slices(recon_ir_LR , insert_to_which_direction = 'x', begin_blank_slice_num = 10, end_blank_slice_num = 40)

    sitk_recon = sitk.GetImageFromArray(recon_ir_LR_save.astype(np.int32))
    sitk_recon.SetSpacing([1,1, 2.828])
    sitk.WriteImage(sitk_recon, os.path.join(main_folder, 'IR', 'IR_corrected_' + str(i)+'.nii.gz'))

    mae_ir,_,rmse_ir,_, ssim_ir = ff.compare(recon_ir_LR, static,cutoff_low = -10, extreme = 1000)

    print('IR results: ', data_loss, mae_ir, rmse_ir, ssim_ir)

    Result.append([i, data_loss, mae_ir, rmse_ir, ssim_ir])
    df = pd.DataFrame(Result, columns = ['step', 'data_loss', 'mae_ir', 'rmse_ir', 'ssim_ir'])
    df.to_excel(os.path.join(main_folder, 'IR', 'loss_record_start_from_PAR.xlsx'), index = False)