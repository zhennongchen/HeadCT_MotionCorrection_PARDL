#!/usr/bin/env python
import numpy as np
import nibabel as nb
import os
from skimage.measure import block_reduce
from skimage import filters
import HeadCT_motion_correction_PAR.motion_simulator.motion_simulation.ct_basic as basic
import HeadCT_motion_correction_PAR.functions_collection as ff
import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.Data_processing as dp

import ct_projector.projector.cupy as ct_projector


cg = Defaults.Parameters()

total_view = 1400  ### 1440 views by default
total_angle = 360
gantry_rotation_time = 500 #unit ms, 500ms by default
K = 12

slice_list = [[5,20]] #[[0,15], [15, 30], [30, 45], [45,60]] #

data_folder = os.path.join(cg.data_dir,'extra/phantom')

patient_list= ff.sort_timeframe(ff.find_all_target_files(['random_*'],data_folder),0,'_')
print(len(patient_list))


for z in range(1376,len(patient_list)):

    p = patient_list[z]
    random_id = os.path.basename(p)
    print('random: ',random_id)

    # make save folder
    save_folder = os.path.join(data_folder,random_id, 'PAR')
    save_folder_sub = os.path.join(save_folder, 'ds')
    ff.make_folder([save_folder, save_folder_sub])

    if os.path.isfile(os.path.join(save_folder_sub,'PARs_ds_crop.nii.gz')) == 1:
        print('done, continue'); continue
  
    # load projection and image and motion parameter:
    if os.path.isfile(os.path.join(p, 'image_data/recon_partial.nii.gz')) == 0:
        print('no image data, skip'); continue

    filename = os.path.join(p, 'projection.nii.gz')
    use_nb = 1
    projection = nb.load(filename).get_fdata()
    projection = projection[:,:,np.newaxis,:]
    print('projection original shape: ', projection.shape)

    if projection.shape[0] < 60:
        print('not enough slices, skip'); continue
    else:
        projection = projection[15:65,...]  #############
        
    img,spacing,affine, header = basic.basic_image_processing(os.path.join(p, 'image_data/recon_partial.nii.gz'), header = True)
    img = img[np.newaxis, ...]

    if os.path.isfile(os.path.join(p,'motion_parameters.npy')) == 1:
        motion_parameters = np.load(os.path.join(p,'motion_parameters.npy'), allow_pickle = True)
        angles = ff.get_angles_zc(total_view, 360, motion_parameters[6])
    else:
        angles = ff.get_angles_zc(total_view, 360, 0)
        print('static, use 0 as SGA')

    print('sga', motion_parameters[6])

    for s in slice_list:
        s1 = s[0]; s2 = s[1]
            
        # prepare
        img_partial = img[:,s1:s2,...]
        projection_partial = projection[s1: s2,...]
        print('image shape: ', img_partial.shape, ' projection shape: ', projection_partial.shape)

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

        # divide sinogram
        sinogram_segments, center_angle_index, num_angles_in_one_segment, segment_indexes = basic.divide_sinogram_new(projection_partial, K , total_view)


        # make PAR
        PAR_collections = basic.make_PAR_new(sinogram_segments, segment_indexes, angles, img_partial[0,...].shape, projector, projector, 'fan')
        PAR_collections = np.rollaxis(PAR_collections,1,4)
        print(PAR_collections.shape)

        # process PAR, second try downsampling:
        PAR_collections_ds = block_reduce(PAR_collections, block_size=(1,2,2,1), func=np.mean)
 

        par_final = np.zeros([25, 128, 128, 15])

        for j in range(0, PAR_collections_ds.shape[0]):
           
            par_final[j,...] = dp.crop_or_pad(PAR_collections_ds[j,...], [128, 128, 15], value = np.min(img))

        
        print(par_final.shape)

        nb.save(nb.Nifti1Image(par_final, affine), os.path.join(save_folder_sub,'PARs_ds_crop.nii.gz'))
    