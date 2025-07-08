#!/usr/bin/env python
import HeadCT_motion_correction_PAR.functions_collection as ff
from HeadCT_motion_correction_PAR.Build_lists import Build_list
import HeadCT_motion_correction_PAR.Defaults as Defaults

import os
import numpy as np
import pandas as pd
import nibabel as nb

cg = Defaults.Parameters()

patient_list = ff.find_all_target_files(['*/*'], os.path.join(cg.data_dir,'simulated_data_3D_spline_6degrees_HR'))

result = []
for p in patient_list:
    patient_id = os.path.basename(os.path.dirname(p))
    patient_subid = os.path.basename(p)

    print(patient_id, patient_subid)

    static = nb.load(os.path.join(cg.data_dir, 'simulated_data_3D_spline_6degrees_HR', patient_id, patient_subid, 'static', 'image_data/recon.nii.gz')).get_fdata()

    hard = nb.load(os.path.join(cg.data_dir,  'simulated_data_3D_spline_6degrees_HR', patient_id, patient_subid, 'random_1/image_data/recon.nii.gz')).get_fdata()

    # easy = nb.load(os.path.join(cg.data_dir,  'simulated_data_3D_spline_6degrees_HR_2', patient_id, patient_subid, 'random_1/image_data/recon.nii.gz')).get_fdata()

    four = nb.load(os.path.join(cg.data_dir,  'simulated_data_3D_spline_HR', patient_id, patient_subid, 'random_1/image_data/recon.nii.gz')).get_fdata()


    hard_mae_all, _, hard_rmse_all, _, hard_ssim_all = ff.compare(hard[:,:,0 + 20 : hard.shape[-1] - 20], static[:,:,0 + 20 : hard.shape[-1] - 20], cutoff_low=-100)
    # easy_mae_all, _, easy_rmse_all, _, easy_ssim_all = ff.compare(easy[:,:,0 + 20 : hard.shape[-1] - 20], static[:,:,0 + 20 : hard.shape[-1] - 20], cutoff_low=-100)
    four_mae_all, _, four_rmse_all, _, four_ssim_all = ff.compare(four[:,:,0 + 20 : hard.shape[-1] - 20], static[:,:,0 + 20 : hard.shape[-1] - 20], cutoff_low=-100)

    result.append([patient_id, patient_subid, hard_mae_all, hard_rmse_all, hard_ssim_all, four_mae_all, four_rmse_all, four_ssim_all])
    df = pd.DataFrame(result, columns = ['patient_id', 'patient_subid', 'hard_mae_all', 'hard_rmse_all', 'hard_ssim_all', 'four_mae_all', 'four_rmse_all', 'four_ssim_all'])
    df.to_excel(os.path.join(cg.data_dir, 'c.xlsx'), index=False)