#!/usr/bin/env python
import numpy as np
import cupy as cp
import nibabel as nb
import os
import pandas as pd
from skimage.measure import block_reduce
import ct_basic as basic
import HeadCT_motion_correction_PAR.functions_collection as ff
import HeadCT_motion_correction_PAR.motion_simulator.transformation as transform
import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.Data_processing as dp

import ct_projector.projector.cupy as ct_projector


cg = Defaults.Parameters()

total_view = 1400  ### 1440 views by default
total_angle = 360
gantry_rotation_time = 500 #unit ms, 500ms by default
K = 12

slice_list = [[0,50]] #[[0,15], [15, 30], [30, 45], [45,60]] #

data_name = 'PAR_3D_spline'
data_folder = os.path.join(cg.data_dir,'simulated_data_3D_spline')
main_save_folder = os.path.join(cg.data_dir,data_name)

patient_list= ff.find_all_target_files(['MO101701M000007/MO001A000009'],data_folder)
print(len(patient_list))


for p in patient_list:

    patient_subid = os.path.basename(p)
    patient_id = os.path.basename(os.path.dirname(p))

    random_folders = ff.find_all_target_files(['random_*'],p)
   

    for r in random_folders:

        random_name = os.path.basename(r)
        print(patient_id, patient_subid, random_name)

        if os.path.isfile(os.path.join(main_save_folder,patient_id, patient_subid, random_name,'slice_0_to_50', 'ds/PARs_ds_crop.nii.gz')) == 1:
            print('done'); continue

        # load projection and image and motion parameter:
        if os.path.isfile(os.path.join(r, 'image_data/recon_partial.nii.gz')) == 0:
            print('no image data, skip'); continue

        filename = os.path.join(r, 'projection.nii.gz')
        use_nb = 1
        projection = nb.load(filename).get_fdata()
        projection = projection[:,:,np.newaxis,:]

        if projection.shape[0] < 60:
            print('not enough slices, skip'); continue
        else:
            projection = projection[10:60,...]  #############
        
        img,spacing,affine, header = basic.basic_image_processing(os.path.join(r, 'image_data/recon_partial.nii.gz'), header = True)
        img = img[np.newaxis, ...]

        if os.path.isfile(os.path.join(r,'motion_parameters.npy')) == 1:
            motion_parameters = np.load(os.path.join(r,'motion_parameters.npy'), allow_pickle = True)
            angles = ff.get_angles_zc(total_view, 360, motion_parameters[6])
        else:
            angles = ff.get_angles_zc(total_view, 360, 0)
            print('static, use 0 as SGA')

        for s in slice_list:
            s1 = s[0]; s2 = s[1]
            print(s1,s2)
            # make save folder
            save_folder = os.path.join(main_save_folder,patient_id, patient_subid, random_name,'slice_'+str(s1)+'_to_'+str(s2))
            ff.make_folder([os.path.dirname(os.path.dirname(os.path.dirname(save_folder))), os.path.dirname(os.path.dirname(save_folder)), os.path.dirname(save_folder), save_folder])
            
            # prepare
            img_partial = img[:,s1:s2,...]
            projection_partial = np.copy(projection)

            # projection_partial = projection[s1: s2,...]
            print('image shape: ', img_partial.shape, ' projection shape: ', projection_partial.shape)
            projector = basic.define_forward_projector(img_partial,spacing,total_view)
            fbp_projector = basic.backprojector(img_partial,spacing)

            # divide sinogram
            sinogram_segments, center_angle_index, num_angles_in_one_segment, segment_indexes = basic.divide_sinogram_new(projection_partial, K , total_view)

            # make PAR
            PAR_collections = basic.make_PAR_new(sinogram_segments, segment_indexes, angles, img_partial[0,...].shape, projector, fbp_projector, 'fan')
            PAR_collections = np.rollaxis(PAR_collections,1,4)
            print(PAR_collections.shape)

            # process PAR, second try downsampling:
            PAR_collections_ds = block_reduce(PAR_collections, block_size=(1,2,2,1), func=np.mean)
            print(PAR_collections_ds.shape)
            save_folder_sub = os.path.join(save_folder, 'ds'); ff.make_folder([save_folder_sub])
            recon_nb = nb.Nifti1Image(PAR_collections_ds,affine, header = header)
            # nb.save(recon_nb, os.path.join(save_folder_sub,'PARs_ds.nii.gz'))

            crop_img = np.zeros([2*K + 1, 128,128 , PAR_collections_ds.shape[-1]])
            for j in range(0,PAR_collections_ds.shape[0]):
                crop_img[j,...] = dp.crop_or_pad(PAR_collections_ds[j,...], [128,128,PAR_collections_ds.shape[-1] ], np.min(PAR_collections_ds[j,...]))
            print('crop final image shape: ' ,crop_img.shape)

            nb.save(nb.Nifti1Image(crop_img, affine, header = header), os.path.join(save_folder_sub,'PARs_ds_crop.nii.gz'))