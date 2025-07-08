# with known MVF, we apply motion to each PAR and get the ground truth (with streaking artifacts) -> for AI ground truth
# use Spatial transformer for the consistency


import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import glob as gb
import nibabel as nb
import math
import pandas as pd
import os
import ct_basic as ct
import HeadCT_motion_correction_PAR.functions_collection as ff
import HeadCT_motion_correction_PAR.motion_simulator.transformation as transform
import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.STN.model_components as compo

cg = Defaults.Parameters()
total_view_num = 1400
K = 12

main_path = os.path.join(cg.data_dir, 'PAR_2D_spline')
data_path = os.path.join(cg.data_dir,'simulated_data_2D_spline')
patient_list = ff.find_all_target_files(['MO101701M00001*/*'],main_path)


for p in patient_list:

    patient_subid = os.path.basename(p)
    patient_id = os.path.basename(os.path.dirname(p))

    random_folders = ff.sort_timeframe(ff.find_all_target_files(['random*'],p),0,'_')
    # random_folders = np.concatenate( [random_folders, ff.find_all_target_files(['static'],p)],axis = 0)

    for random in random_folders:
        random_name = os.path.basename(random)

        # load PAR:
        # file_name = ff.find_all_target_files(['*/original/PARs_original.np*','*/ds/PARs_ds_crop.np*' ], random)
        file_name = ff.find_all_target_files(['slice_30_to_45/ds/PARs_ds_crop.np*' ], random)
        
        for f in file_name:
            slice_num = os.path.basename(os.path.dirname(os.path.dirname(f)))
            PAR_type = os.path.basename(os.path.dirname(f))
            print(patient_id, patient_subid, random_name, slice_num, PAR_type)

            if os.path.isfile(os.path.join(random, slice_num, 'image_by_PAR_true_MVF','image_by_PAR_'+PAR_type+'.nii.gz')) == 1:
                print('done'); continue
        
            # load PAR
            par = np.load(f, allow_pickle = True)
            par_num = par.shape[0]
            print(par_num)

            # load motion parameters
            parameters = os.path.join(data_path, patient_id, patient_subid,random_name, 'motion_parameters.npy' )
            parameters = np.load(parameters, allow_pickle = True)

            tx= parameters[0,:][0]
            ty= parameters[2,:][0]
            r = parameters[4,:][0]

            indexes,_,_ = ct.divide_sinogram_indexes(K, total_view_num)

            spline_x = transform.interp_func(np.linspace(0,1,5),np.asarray(tx))
            spline_y = transform.interp_func(np.linspace(0,1,5), np.asarray(ty))
            spline_r = transform.interp_func(np.linspace(0,1,5), np.asarray(r))

            tx_25 = [ spline_x(indexes[l][1]/ total_view_num) for l in range(0,indexes.shape[0])]
            ty_25 = [ spline_y(indexes[l][1]/total_view_num) for l in range(0,indexes.shape[0])]
            r_25 = [ spline_r(indexes[l][1]/ total_view_num) for l in range(0,indexes.shape[0])]

            # apply motion parameters to each PAR
            d = [2 if PAR_type[0:2] == 'ds' else 1][0]

            # use STN instead
            par_stn = np.copy(par)
            for j in range(0,par.shape[0]):
                I = par[j,:,:,:]
                I = np.expand_dims(I[np.newaxis,...],axis = -1)
                I = np.transpose(I,[3,0,1,2,4])
                rr = float(r_25[j][0])
                translation = np.array([[1,0, -ty_25[j][0] / par.shape[2] * 2/ d], [0,1,-tx_25[j][0]/par.shape[1] * 2 /d ], [0,0,1]])  # STN has opposite coordinate compared with ours
                rotation = np.array([[math.cos(rr), -math.sin(rr), 0 ], [math.sin(rr), math.cos(rr), 0], [0,0,1]]) # STN has opposite rotation compared with ours

                transformation_matrix = np.dot(rotation, translation)
                matrix = transformation_matrix[0:2,:]
                
                a = [compo.BilinearInterpolation(par.shape[1],par.shape[2])([ii,matrix]) for ii in I]
                a = np.transpose(np.stack(a,axis = 0),[1,2,3,0,4])
                par_stn[j,:,:,:] = np.squeeze(a)
            par_stn = np.sum(par_stn,axis = 0) / par_num
            print(par_stn.shape)
    
            save_folder = os.path.join(random, slice_num, 'image_by_PAR_true_MVF')
            ff.make_folder([save_folder])

            # save
            if PAR_type[0:2] != 'ds':
                ii = nb.load(os.path.join(data_path,patient_id,patient_subid,random_name,'image_data/recon_partial.nii.gz'))
            else:
                ii = nb.load(os.path.join(data_path,patient_id,patient_subid,random_name,'image_data/recon_partial_ds.nii.gz'))
            affine = ii.affine; header = ii.header
            nb.save(nb.Nifti1Image(par_stn, affine, header = header), os.path.join(save_folder,'image_by_PAR_'+PAR_type+'.nii.gz'))



# our transformation method
            # par_c = np.copy(par)
            # for j in range(0,par.shape[0]):
            #     I = par[j,...]
            #     _,_,_,transformation_matrix = transform.generate_transform_matrix([-tx_25[j]/d, -ty_25[j]/d,0],[0,0, -r_25[j]],[1,1,1],I.shape)
            #     transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, I.shape)
            #     img_new = transform.apply_affine_transform(I, transformation_matrix,3, cval = np.min(par))
            #     par_c[j,...] = img_new

            # par_c = np.sum(par_c,axis=0) / 25