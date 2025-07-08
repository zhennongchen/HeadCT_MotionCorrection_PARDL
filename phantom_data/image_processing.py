import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.Data_processing as dp
import HeadCT_motion_correction_PAR.functions_collection as ff

import argparse
import os
import numpy as np
import nibabel as nb
from skimage.measure import block_reduce

cg = Defaults.Parameters()


# image processing: save partial volume and downsample
# image_list = ff.find_all_target_files(['study_*/simulated_imgs_resampled/recon_motion_ibc_idr.nii.gz'],os.path.join(cg.data_dir,'phantom_data'))
# # image_list = ff.find_all_target_files(['recon_resampled/recon_ibc_idr.nii.gz'], '/mnt/mount_zc_NAS/head_phantom_raw/processed/study_4/scan4/processed_0')

# slice_range = [45,95] #[55, 105]
# for i in image_list:

#     print(i)
#     if os.path.isfile(os.path.join(os.path.dirname(i),  'recon_motion_ibc_idr_partial.nii.gz')) == 1:
#         print('done')
#         continue

#     img = nb.load(i).get_fdata()

#     img = img[:,:,slice_range[0]:slice_range[1]]
#     # img = dp.crop_or_pad(img, [256, 256, img.shape[-1]], value = np.min(img))

#     print(img.shape)

#     nb.save(nb.Nifti1Image(img.astype(np.float32), nb.load(i).affine), os.path.join(os.path.dirname(i),  'recon_motion_ibc_idr_partial.nii.gz'))
    


# PAR
image_list = ff.sort_timeframe(ff.find_all_target_files(['study_20/PAR/resampled/*.nii.gz'],os.path.join(cg.data_dir,'phantom_data')),2,'_')

par_final = np.zeros([25, 128, 128, 15])

for i in range(0, image_list.shape[0]):
    print(image_list[i])


    patient_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_list[i]))))

    img = nb.load(image_list[i]).get_fdata()

    img= block_reduce(img, block_size=(2,2,1), func=np.mean)

    img = dp.crop_or_pad(img, [128, 128, img.shape[-1]], value = np.min(img))

    par_final[i,...] = img

print(par_final.shape)
ff.make_folder([os.path.join(cg.data_dir,'phantom_data', patient_id, 'PAR', 'ds')])
nb.save(nb.Nifti1Image(par_final, nb.load(image_list[i]).affine), os.path.join(cg.data_dir,'phantom_data', patient_id, 'PAR', 'ds', 'PARs_ds.nii.gz'))