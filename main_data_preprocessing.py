import Defaults
import Data_processing as dp
import functions_collection as ff
from Build_lists import Build_list

import argparse
import os
import numpy as np
import nibabel as nb
import tensorflow as tf
import shutil


cg = Defaults.Parameters()

# image processing: save partial volume and downsample
# image_list = ff.find_all_target_files(['MO101701M00000*/*/random_*/image_data/recon.nii.gz'],os.path.join(cg.data_dir,'simulated_data_3D_spline_6degrees'))
# dp.save_partial_volumes(image_list,'recon_partial.nii.gz',slice_range = [10,60])
# image_list = ff.find_all_target_files(['*/*/*/image_data/recon_partial.nii.gz'],os.path.join(cg.data_dir,'simulated_data_2D_spline'))
# dp.downsample_crop_image(image_list,'recon_partial_ds.nii.gz', factor = [2,2,1])


    
# image processing: crop and pad
image_list = ff.find_all_target_files(['MO101701M00002*/*/random_*/all_slices/PARs_ds_crop_anneal.nii.gz', 'MO101701M00003*/*/random_*/all_slices/PARs_ds_crop_anneal.nii.gz'],os.path.join(cg.data_dir,'PAR_3D_spline_6degrees_HR'))
for i in image_list:
    print(i)
    # if os.path.isfile(os.path.join(os.path.dirname(i),'PARs_slice_35.nii.gz')) == 1:
    #     print('done')
    #     continue
    img = nb.load(i)
    
    new_img = np.copy(img.get_fdata()[:,:,:,0:25]).astype(np.int32)
    nb.save(nb.Nifti1Image(new_img, img.affine),os.path.join(os.path.dirname(i) ,'PARs_slice_55.nii.gz'))

    new_img = np.copy(img.get_fdata()[:,:,:,25:50]).astype(np.int32)
    nb.save(nb.Nifti1Image(new_img, img.affine),os.path.join(os.path.dirname(i) ,'PARs_slice_115.nii.gz'))

    new_img = np.copy(img.get_fdata()[:,:,:,50:75]).astype(np.int32)
    nb.save(nb.Nifti1Image(new_img, img.affine),os.path.join(os.path.dirname(i) ,'PARs_slice_175.nii.gz'))

    os.remove(i)
    
    

    # new_img = np.zeros([cg.par_num, 128,128 , cg.dim[2]])
    # for j in range(0,img.shape[0]):
    #     new_img[j,...] = dp.crop_or_pad(img.get_fdata()[j,:,:,:], [128,128, cg.dim[2]], np.min(img.get_fdata()[j,...]))
    # print(new_img.shape)

    # ff.make_folder([os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(i))), 'slice_0_to_50')])
    # nb.save(nb.Nifti1Image(new_img, img.affine),os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(i))), 'slice_0_to_50' ,'PARs_ds_crop.nii.gz'))
    
   


# anneal slices in PAR
# image_list = ff.find_all_target_files(['MO101701M00000*/*/random_*/all_slices/ds/PARs_ds_crop.nii.gz'],os.path.join(cg.data_dir,'PAR_3D_spline_HR'))
# for i in image_list:
#     print(i)

#     if os.path.isfile(os.path.join(os.path.dirname(i), 'PARs_ds_crop_anneal.nii.gz')) == 1:
#         continue

#     img_file = nb.load(i)
#     img = img_file.get_fdata()

#     new_par = np.zeros([25,128,128,75])
#     new_par[:,:,:, 0:25] = img[:,:,:, 55:80]
#     new_par[:,:,:,25:50] = img[:,:,:, 115:140]
#     new_par[:,:,:, 50:75] = img[:,:,:, 175:200]

#     nb.save(nb.Nifti1Image(new_par.astype(np.int32), img_file.affine), os.path.join(os.path.dirname(i), 'PARs_ds_crop_anneal.nii.gz'))
