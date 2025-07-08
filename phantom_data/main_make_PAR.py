#!/usr/bin/env python
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import glob as gb
import nibabel as nb
import math
import os
import pandas as pd
from skimage.measure import block_reduce
from skimage import filters
import HeadCT_motion_correction_PAR.motion_simulator.motion_simulation.ct_basic as basic
import HeadCT_motion_correction_PAR.functions_collection as ff
import HeadCT_motion_correction_PAR.motion_simulator.transformation as transform
import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.Data_processing as processing

import CTProjector.src.ct_projector.projector.numpy as ct_projector
import ct_projector.projector.cupy.fan_equiangular as ct_fan
import ct_projector.projector.numpy as numpy_projector
import ct_projector.projector.numpy.fan_equiangluar as numpy_fan

cg = Defaults.Parameters()

total_view = 1400  
total_angle = 360
gantry_rotation_time = 500 #unit ms, 500ms by default
K = 12

slice_list = [[5,20]] #[[0,15], [15, 30], [30, 45], [45,60]] #

data_name = 'phantom_data'
data_folder = os.path.join(cg.data_dir,'phantom_data')

patient_list= ff.find_all_target_files(['study_*'],data_folder)
print(len(patient_list))


for p in patient_list:

    patient_id = os.path.basename(p)
    print(patient_id)

    save_folder = os.path.join(p, 'PAR', 'original'); ff.make_folder([os.path.dirname(save_folder), save_folder])

    if os.path.isfile(os.path.join(save_folder, 'PAR_10.nii.gz')) == 1:
        print('done');continue

    # load projection and image and motion parameter:
    if os.path.isfile(os.path.join(p, 'simulated_imgs_raw/recon_motion.nii.gz')) == 0:
        print('no image data, skip'); continue

    filename = os.path.join(p, 'projection.nii.gz')
    use_nb = 1
    projection = nb.load(filename).get_fdata()
    projection = projection[45:95,...]
    projection = projection[:,:,np.newaxis,:]
    print(projection.shape)

    img,spacing,affine, header = basic.basic_image_processing(os.path.join(p, 'simulated_imgs_raw/recon_motion.nii.gz'), header = True)
    img = img[45:95,...]
    img = img[np.newaxis, ...]

    angles = ff.get_angles_zc(1440, 360, 0)

    for s in slice_list:
        s1 = s[0]; s2 = s[1]
        print(s1,s2)
    
        # prepare
        img_partial = img[:,s1:s2,...]
        projection_partial = projection[s1: s2,...]
        print('image shape: ', img_partial.shape, ' projection shape: ', projection_partial.shape)
        projector = ct_projector.ct_projector()
        projector.from_file('./projector_fan_scanner.cfg')
        projector.nv = 1
        projector.nz = 1
        projector.dv = 2.828
        projector.dz = 2.828
        

        # divide sinogram
        sinogram_segments, center_angle_index, num_angles_in_one_segment, segment_indexes = basic.divide_sinogram_new(projection_partial, K , total_view, fill_out = False)
        
        # make PAR
        PAR_collections = basic.make_PAR_new(sinogram_segments, segment_indexes, angles, img_partial[0,...].shape, projector, projector, 'fan')
        PAR_collections = np.rollaxis(PAR_collections,1,4)
        print(PAR_collections.shape)

    # save
    for n in range(PAR_collections.shape[0]):
        nb.save(nb.Nifti1Image(PAR_collections[n,...], affine), os.path.join(save_folder,'PAR_' + str(n) + '.nii.gz'))
