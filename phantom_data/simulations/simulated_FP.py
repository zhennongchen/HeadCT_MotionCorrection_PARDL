import numpy as np
import cupy as cp
import os
import nibabel as nb
import HeadCT_motion_correction_PAR.motion_simulator.motion_simulation.ct_basic as ct
import pandas as pd
import HeadCT_motion_correction_PAR.functions_collection as ff
import HeadCT_motion_correction_PAR.motion_simulator.transformation as transform
import HeadCT_motion_correction_PAR.Defaults as Defaults

import CTProjector.src.ct_projector.projector.cupy as ct_projector
from HeadCT_motion_correction_PAR.Build_lists import Build_list


cg = Defaults.Parameters()

# # define patients
data_folder = '/mnt/mount_zc_NAS/head_phantom_raw/processed/study_4/scan4/processed_0/recon_cropped'
save_folder = os.path.join(cg.data_dir,'extra','phantom')


# define some basic parameters and image
CP_num = 5
geometry = 'fan'
total_view = 1400  ### 2340 views by default
gantry_rotation_time = 500 #unit ms, 500ms by default
view_increment = 50 # increment in gantry views

count  = 0
for i in range(601, 1101 ):

    sub_folder = os.path.join(save_folder, 'random_' + str(i))
    ff.make_folder([sub_folder])

    if os.path.isfile(os.path.join(sub_folder,'image_data','recon.nii.gz')) == 1:
        print('done, skip')
        continue

    img,spacing,img_affine = ct.basic_image_processing(os.path.join(data_folder, 'recon.nii.gz'))
    print(spacing, img.shape)

    img = img[np.newaxis, ...]
    projector = ct_projector.ct_projector()
    projector.from_file('./projector_fan_scanner.cfg')

    projector.nview = 1400
    projector.nx = 256
    projector.ny = 256
    projector.nv = 1
    projector.nz = 1
    projector.dx = 1.0
    projector.dy = 1.0
    projector.dz = 2.828000068664551
    projector.dsd = 596.0
    projector.dso = 366.0
    projector.du = 1.145
    projector.dv = 2.828000068664551
 
    cp.cuda.Device(0).use()
    ct_projector.set_device(0)

    t = np.linspace(0, gantry_rotation_time, CP_num, endpoint=True)

    # amplitude_t_mm = transform.motion_control_point_generation(2, CP_num, 6, 3, 2, print_result = True)
    # amplitude_tx_mm = amplitude_t_mm[:,0]; amplitude_tz_mm = amplitude_t_mm[:,1]
    # # amplitude_tx_mm = transform.motion_control_point_generation(1, CP_num, 6,3, 2, print_result = True)[:,0]
    # # amplitude_tz_mm = transform.motion_control_point_generation(1, CP_num, 5.8, 2.828, 2, print_result = True)[:,0]

    # amplitude_ty_mm = [0,0,0,0,0]
    # amplitude_rx_degree = transform.motion_control_point_generation(1, CP_num, 5,2.5, 2, print_result = True)[:,0]
    # amplitude_rz_degree = transform.motion_control_point_generation(1, CP_num, 5, 2.5, 2, print_result = True)[:,0]
    # amplitude_ry_degree = [0,0,0,0,0]


    amplitude_tx_mm = [0,0,0,0,0]
    amplitude_tz_mm = [0,0,0,0,0]
    amplitude_ty_mm = [0,0,0,0,0]
    amplitude_rx_degree = [0,0,0,0,0]
    amplitude_rz_degree = [0. , 1.3 ,3. , 2.5 ,2. ]
    amplitude_ry_degree = [0,0,0,0,0]

    # spline fit
    spline_tx = transform.interp_func(t, np.asarray([i/1.0 for i in amplitude_tx_mm]))
    spline_ty = transform.interp_func(t, np.asarray([i/1.0 for i in amplitude_ty_mm]))
    spline_tz = transform.interp_func(t, np.asarray([i/2.828 for i in amplitude_tz_mm]))
    spline_rx = transform.interp_func(t,np.asarray([i / 180 * np.pi for i in amplitude_rx_degree]))
    spline_ry = transform.interp_func(t,np.asarray([i / 180 * np.pi for i in amplitude_ry_degree]))
    spline_rz = transform.interp_func(t,np.asarray([i / 180 * np.pi for i in amplitude_rz_degree]))
    

    sga =  0#int(np.random.uniform(0,90))
    angles = ff.get_angles_zc(total_view, 360  ,sga)

    parameter_file = os.path.join(sub_folder, 'motion_parameters.txt')
    # ff.txt_writer(parameter_file,[t.tolist(),amplitude_tx_mm, amplitude_ty_mm, amplitude_tz_mm, amplitude_rx_degree, amplitude_ry_degree, amplitude_rz_degree, [sga],[total_view],[gantry_rotation_time]],['time_points','translation_x_CP','translation_y_CP','translation_z_CP', 'rotation_x_CP', 'rotation_y_CP','rotation_z_CP','starting_gantry_angle', 'total_projection_view','gantry_rotation_time(ms)'])
    parameter_file = os.path.join(sub_folder,'motion_parameters.npy')
    # np.save(parameter_file, np.array([[amplitude_tx_mm],[amplitude_ty_mm],[amplitude_tz_mm],[amplitude_rx_degree],[amplitude_ry_degree],[amplitude_rz_degree],[sga], [t], [total_view], [gantry_rotation_time]], dtype=object))
        

    projection = ct.fp_w_spline_motion_model(img, projector, angles,  spline_tx, spline_ty, spline_tz, spline_rx,  spline_ry, spline_rz, 
                                        geometry, total_view = total_view, gantry_rotation_time = gantry_rotation_time, slice_num = None, increment = view_increment, order = 3)
    projection_save_version = nb.Nifti1Image(projection[:,:,0,:], img_affine)
    # nb.save(projection_save_version, os.path.join(sub_folder,'projection.nii.gz'))

    # generate backprojection
    recon = ct.filtered_backporjection(projection,angles,projector,projector, geometry, back_to_original_value=True)

    # save recon
    recon_nb_image = np.rollaxis(recon,0,3)
    ff.make_folder([os.path.join(sub_folder,'image_data')])
    # nb.save(nb.Nifti1Image(recon_nb_image,img_affine), os.path.join(sub_folder,'image_data','recon.nii.gz'))

    img = nb.load(os.path.join(data_folder, 'recon.nii.gz')).get_fdata()

    mae, mse, rmse,ssim = ff.compare(recon_nb_image, img, cutoff_low = -10 )
    print(mae, mse, rmse, ssim)



