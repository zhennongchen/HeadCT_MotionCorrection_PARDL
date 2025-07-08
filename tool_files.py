import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.functions_collection as ff
from HeadCT_motion_correction_PAR.Build_lists import Build_list
import Diffusion_models.Data_processing as dp

import os
import numpy as np
import nibabel as nb
import pandas as pd
import shutil
import SimpleITK as sitk

cg = Defaults.Parameters()

# resample pixel dimension!
# patient_list = ff.find_all_target_files(['*/*/random_16','*/*/random_17','*/*/random_18','*/*/random_19','*/*/random_20'], os.path.join(cg.data_dir, 'simulated_data_portable_CT_new'))

# for patient in patient_list:
#     patient_id = os.path.basename(os.path.dirname(os.path.dirname(patient)))
#     patient_subid = os.path.basename(os.path.dirname(patient))
#     random_id = os.path.basename(patient)

#     if os.path.isfile(os.path.join(patient,'image_data/recon_resample.nii.gz')):
#         print('did this')
#         continue

#     print(patient_id, patient_subid, random_id)
    
#     hr = nb.load(os.path.join(patient, 'image_data/recon.nii.gz'))
#     lr = nb.load(os.path.join(cg.data_dir, 'simulated_data_3D_spline',patient_id, patient_subid, 'random_1', 'image_data/recon_partial.nii.gz'))

#     # for data
#     lr_dim = lr.header.get_zooms()[:3]
#     hr_dim = hr.header.get_zooms()[:3]

#     # for phantom:
#     # lr_dim = [1,1, 2.5]

#     # for data:
#     hr_resample = dp.resample_nifti(hr, order=3,  mode = 'nearest',  cval = np.min(hr.get_fdata()), in_plane_resolution_mm=lr_dim[0], slice_thickness_mm=lr_dim[-1])
#     hr_resample = nb.Nifti1Image(hr_resample.get_fdata(), affine=hr_resample.affine, header=hr_resample.header)

#     # for phantom
#     # hr_resample = nb.Nifti1Image(hr.get_fdata()[50:300, 30:280,:], affine=hr.affine, header=hr.header)  

#     nb.save(hr_resample, os.path.join(patient,'image_data/recon_resample.nii.gz'))

# # convert npy to niigz
# affine = nb.load(os.path.join(cg.data_dir,'simulated_data_2D_spline','MO101701M000001/MO001A000001/random_1/image_data/recon.nii.gz')).affine
# file_list = ff.find_all_target_files(['*/*/*'],os.path.join(cg.data_dir,'simulated_data_2D_spline'))
# for f in file_list:
#     npy_file = os.path.join(f, 'projection.npy')
#     if os.path.isfile(npy_file) == 1:
#         print(f)
#         a = np.load(npy_file,allow_pickle = True)
#         a = a[:,:,0,:]
#         print(a.shape)
#         nb.save(nb.Nifti1Image(a,affine),os.path.join(f,'projection.nii.gz'))
#         os.remove(npy_file)


# check image dimension
# file_list = ff.find_all_target_files(['recon_try.nii.gz'],os.path.join(cg.data_dir,'phantom_data/study_3/simulated_imgs_resampled'))

# for p in file_list:
#     a = sitk.ReadImage(p)
#     b = sitk.GetArrayFromImage(a)
#     b = b.astype(np.int32)
#     print(b.shape, a.GetSpacing())

#     sitk_recon = sitk.GetImageFromArray(b)
#     sitk_recon.SetSpacing(a.GetSpacing())
#     sitk.WriteImage(sitk_recon, os.path.join(os.path.dirname(p), 'recon_try1.nii.gz'))

    

# make snapshot
# patient_list = ff.find_all_target_files(['*/*'],os.path.join(cg.predict_dir,'CNN_DResNet_2/images'))
# batch_list = [0,1,2,3,4]
# for p in patient_list:
    
#     patient_subid = os.path.basename(p)
#     patient_id = os.path.basename(os.path.dirname(p))
    

#     folders = ff.sort_timeframe(ff.find_all_target_files(['random_*'],p),0,'_')
#     if len(folders) == 0:
#         folders = ff.find_all_target_files(['real_data'],p)
    
#     for f in folders:
#         random_num  = os.path.basename(f)
#         print(patient_id,patient_subid,random_num)

#         for batch in batch_list:

#             img = nb.load(os.path.join(f,'batch_'+str(batch), 'pred.nii.gz')).get_fdata()
  
#             save_folder = os.path.join(cg.picture_dir,'snapshot_CNN_DResNet_2',patient_id, patient_subid, random_num, 'batch_'+str(batch))
            
#             ff.make_folder([os.path.dirname(os.path.dirname(os.path.dirname(save_folder))), os.path.dirname(os.path.dirname(save_folder)), os.path.dirname(save_folder), save_folder])
#             # if os.path.isfile(os.path.join(save_folder,'pred_0.png')) == 1:
#             #     print('done')
#             #     continue

#             for jj in range(0,img.shape[-1]):
#                 if jj%2 == 0:
#                     ff.save_grayscale_image(np.flip(ff.set_window(img[:,:,jj].T,500,2000),0), os.path.join(save_folder,'pred_bone_'+str(jj)+'.png'))
#                     ff.save_grayscale_image(np.flip(ff.set_window(img[:,:,jj].T,50,100),0), os.path.join(save_folder,'pred_brain_'+str(jj)+'.png'))



# patient_list = ff.find_all_target_files(['*/*'],os.path.join(cg.data_dir,'simulated_data_new'))
# print(len(patient_list))
# for p in patient_list:
#     patient_subid = os.path.basename(p)
#     patient_id = os.path.basename(os.path.dirname(p))
#     print(patient_id,patient_subid)

#     folders = ff.sort_timeframe(ff.find_all_target_files(['random_*'],p),0,'_')
    
#     for j in folders:
#         random_num = os.path.basename(j)

#         img = nb.load(os.path.join(j,'image_data/recon_partial.nii.gz')).get_fdata()
  
#         save_folder = os.path.join(cg.picture_dir,'snapshot_simulated_new',patient_id,patient_subid,random_num)
#         ff.make_folder([os.path.dirname(os.path.dirname(save_folder)), os.path.dirname(save_folder), save_folder])
#         # if os.path.isfile(os.path.join(save_folder,'recon_0.png')) == 1:
#         #     print('done')
#         #     continue

#         for jj in range(0,img.shape[-1]):
#             if jj%2 == 0:
#                 ff.save_grayscale_image(np.flip(ff.set_window(img[:,:,jj].T,500,2000),0), os.path.join(save_folder,'simulated_bone_'+str(jj)+'.png'))
#                 ff.save_grayscale_image(np.flip(ff.set_window(img[:,:,jj].T,50,100),0), os.path.join(save_folder,'simulated_brain_'+str(jj)+'.png'))
 


# patient_list = ff.find_all_target_files(['*/*'],os.path.join(cg.data_dir,'raw_data/nii-images/thin_slice'))
# print(len(patient_list))
# for p in patient_list:
#     patient_subid = os.path.basename(p)
#     patient_id = os.path.basename(os.path.dirname(p))
#     print(pazwtient_id,patient_subid)

#     img = nb.load(os.path.join(p,'img-nii-1.5/img_partial.nii.gz')).get_fdata()
  
#     save_folder = os.path.join(cg.picture_dir,'snapshot_motion_free',patient_id,patient_subid)
#     ff.make_folder([os.path.dirname(save_folder),save_folder])
#     # if os.path.isfile(os.path.join(save_folder,'recon_0.png')) == 1:
#     #     print('done')
#     #     continue

#     for jj in range(0,img.shape[-1]):
#         if jj%2 == 0:
#             ff.save_grayscale_image(np.flip(ff.set_window(img[:,:,jj].T,500,2000),0), os.path.join(save_folder,'image_bone_'+str(jj)+'.png'))
#             ff.save_grayscale_image(np.flip(ff.set_window(img[:,:,jj].T,50,100),0), os.path.join(save_folder,'image_brain_'+str(jj)+'.png'))


# delete files
# files = ff.find_all_target_files(['*/*/random_6','*/*/random_7','*/*/random_8','*/*/random_9','*/*/random_10'],'/mnt/camca_NAS/Portable_CT_Data/simulations_202404/simulated_all_motion_v1')

# for i in range(0,files.shape[0]):
#     # os.remove(files[i])
#     shutil.rmtree(files[i])

 

# # check image
# file_list = ff.find_all_target_files(['*/*/random_*'],os.path.join('/mnt/camca_NAS/Portable_CT_data/simulations_202404','simulated_all_motion_v2'))
# for f in file_list:
#     random_name = os.path.basename(f)
#     patient_subid = os.path.basename(os.path.dirname(f))
#     patient_id = os.path.basename(os.path.dirname(os.path.dirname(f)))
#     # print(patient_id,patient_subid,random_name)
#     if os.path.isfile(os.path.join(f,'image_data/recon.nii.gz')) == 0:
#         continue
#     img = nb.load(os.path.join(f,'image_data/recon.nii.gz')).get_fdata()

#     for slice_num in range(10,img.shape[-1]- 10):
#         ii = img[:,:,slice_num]
#         if np.all(abs(ii - -1024) <= 100):
#             print('WRRRRONG ',patient_id,patient_subid,random_name, slice_num)
#             os.remove(os.path.join(f,'image_data/recon.nii.gz'))
#             if os.path.isfile(os.path.join(f,'image_data/recon_resample.nii.gz')) == 1:
#                 os.remove(os.path.join(f,'image_data/recon_resample.nii.gz'))
#             break

    
# copy files/folders
des_folder = os.path.join('/mnt/camca_NAS/Portable_CT_data/simulations_202404','simulated_all_motion_v2')
ori_folder = os.path.join('/mnt/camca_NAS/Portable_CT_data/simulations_202404','simulated_all_motion_v3')
folders = ff.find_all_target_files(['*/*/random_*'],ori_folder)
for f in folders:
    patient_id = os.path.basename(os.path.dirname(os.path.dirname(f)))
    patient_subid = os.path.basename(os.path.dirname(f))
    random_name = os.path.basename(f)
    print(patient_id, patient_subid , random_name)

    ff.make_folder([os.path.join(des_folder, patient_id), os.path.join(des_folder, patient_id, patient_subid), os.path.join(des_folder, patient_id, patient_subid, random_name)])

    des = os.path.join(des_folder, patient_id, patient_subid,  random_name, 'motion_parameters.npy')

    if os.path.isdir(des) == 0:
        shutil.copyfile(os.path.join(ori_folder, patient_id, patient_subid,random_name,  'motion_parameters.npy'), des)

    des = os.path.join(des_folder, patient_id, patient_subid,random_name,  'motion_parameters.txt')

    if os.path.isdir(des) == 0:
        shutil.copyfile(os.path.join(ori_folder, patient_id, patient_subid, random_name, 'motion_parameters.txt'), des)
    




# # zip folder
# # shutil.make_archive(os.path.join(cg.nas_dir,'picture_collections/snapshot_CNN_DResNet_2'), 'zip', os.path.join(cg.nas_dir,'picture_collections/snapshot_CNN_DResNet_1'))
# # shutil.make_archive(os.path.join(cg.nas_dir,'picture_collections/snapshot_motion_free'), 'zip', os.path.join(cg.nas_dir,'picture_collections/snapshot_motion_free'))
# # shutil.make_archive(os.path.join(cg.nas_dir,'picture_collections/snapshot_simulated_new'), 'zip', os.path.join(cg.nas_dir,'picture_collections/snapshot_simulated'))


    

    