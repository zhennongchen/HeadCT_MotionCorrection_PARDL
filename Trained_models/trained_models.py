import os 
import numpy as np
import HeadCT_motion_correction_PAR.Defaults as Defaults

cg = Defaults.Parameters()

class trained_models():
    def __init__(self):
        self.main = cg.model_dir

    # 3D motion all slices 6 degree of freedom
    def CNN_3D_motion_6degrees(self):
        models = [os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_0/model-025.hdf5'), 
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_0/model-033.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_0/model-034.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_0/model-038.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_0/model-042.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_1/model-018.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_1/model-022.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_1/model-025.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_1/model-033.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_1/model-036.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_1/model-037.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_2/model-009.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_2/model-024.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_2/model-027.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_2/model-032.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_2/model-036.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees/models/batch_2/model-037.hdf5'),
        ]

        return models

    def CNN_3D_motion_6degrees_thin_slice(self):
        models = [os.path.join(self.main, 'CNN_3D_motion_6degrees_thin_slice/models/batch_0/model-078.hdf5'), 
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_thin_slice/models/batch_0/model-083.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_thin_slice/models/batch_0/model-053.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_thin_slice/models/batch_0/model-047.hdf5'), 
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_thin_slice/models/batch_0/model-041.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_thin_slice/models/batch_0/model-045.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_thin_slice/models/batch_1/model-100.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_thin_slice/models/batch_1/model-085.hdf5'),

        ]

        return models

    # ablation study: 7 control points
    def CNN_3D_motion_6degrees_ablation_7CP(self):
        models = [os.path.join(self.main, 'CNN_3D_motion_6degrees_ablation_7CP/models/batch_0/model-038.hdf5'), 
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_ablation_7CP/models/batch_0/model-034.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_ablation_7CP/models/batch_0/model-041.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_ablation_7CP/models/batch_0/model-042.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_ablation_7CP/models/batch_0/model-037.hdf5'),

                  os.path.join(self.main, 'CNN_3D_motion_6degrees_ablation_7CP/models/batch_1/model-052.hdf5'), 
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_ablation_7CP/models/batch_1/model-046.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_ablation_7CP/models/batch_1/model-038.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_ablation_7CP/models/batch_1/model-041.hdf5'),
                  os.path.join(self.main, 'CNN_3D_motion_6degrees_ablation_7CP/models/batch_1/model-053.hdf5'),]
        return models
    

    # 3D motion all slices (4 degrees)
    def CNN_3D_motion_allslices(self):
        tx_models = [os.path.join(self.main, 'CNN_3D_motion_allslices/models/batch_0/model-075.hdf5'), 
                     os.path.join(self.main, 'CNN_3D_motion_allslices/models/batch_1/model-081.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_allslices/models/batch_2/model-053.hdf5'),
        ]

        tz_models = [os.path.join(self.main, 'CNN_3D_motion_allslices/models/batch_0/model-070.hdf5'), 
                     os.path.join(self.main, 'CNN_3D_motion_allslices/models/batch_1/model-088.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_allslices/models/batch_2/model-081.hdf5'),
        ]

        rx_models = [os.path.join(self.main, 'CNN_3D_motion_allslices/models/batch_0/model-069.hdf5'), 
                     os.path.join(self.main, 'CNN_3D_motion_allslices/models/batch_1/model-094.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_allslices/models/batch_2/model-075.hdf5'),
        ]

        rz_models = [os.path.join(self.main, 'CNN_3D_motion_allslices/models/batch_0/model-071.hdf5'), 
                     os.path.join(self.main, 'CNN_3D_motion_allslices/models/batch_1/model-093.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_allslices/models/batch_2/model-080.hdf5'),
        ]

        return tx_models, tz_models, rx_models, rz_models
    
    
    # 3D motion slice 5-20
    def CNN_3D_motion_slice_5_to_20(self):
        tx_models = [os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_0/model-023.hdf5'), # lowest val_tx_loss
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_1/model-032.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_2/model-025.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_3/model-022.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_4/model-038.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_0/model-048.hdf5'), # lowest val_loss
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_1/model-046.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_2/model-046.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_3/model-048.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_4/model-037.hdf5'),
                  ]

        tz_models = [os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_0/model-042.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_1/model-052.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_2/model-049.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_3/model-053.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_4/model-036.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_0/model-048.hdf5'), # lowest val_loss
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_1/model-046.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_2/model-046.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_3/model-048.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_4/model-037.hdf5'),
                  ]

        rx_models = [os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_0/model-038.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_1/model-050.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_2/model-047.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_3/model-047.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_4/model-032.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_0/model-048.hdf5'), # lowest val_loss
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_1/model-046.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_2/model-046.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_3/model-048.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_4/model-037.hdf5'),
                 ]
        
        rz_models = [os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_0/model-038.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_1/model-052.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_2/model-029.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_3/model-042.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_4/model-038.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_0/model-048.hdf5'), # lowest val_loss
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_1/model-046.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_2/model-046.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_3/model-048.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice5-20/models/batch_4/model-037.hdf5'),
                 ]

        return tx_models, tz_models, rx_models, rz_models
    

    # 3D motion slice 20-35
    def CNN_3D_motion_slice_20_to_35(self):
        tx_models = [os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_0/model-050.hdf5'), # lowest val_tx_loss
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_1/model-033.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_2/model-064.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_0/model-050.hdf5'), # lowest val_loss
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_1/model-066.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_2/model-064.hdf5'),
                  ]

        tz_models = [os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_0/model-049.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_1/model-043.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_2/model-044.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_0/model-050.hdf5'), # lowest val_loss
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_1/model-066.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_2/model-064.hdf5'),
                  ]

        rx_models = [os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_0/model-050.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_1/model-062.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_2/model-043.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_0/model-050.hdf5'), # lowest val_loss
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_1/model-066.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_2/model-064.hdf5'),
                 ]
        
        rz_models = [os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_0/model-055.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_1/model-064.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_2/model-063.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_0/model-050.hdf5'), # lowest val_loss
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_1/model-066.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_slice20-35/models/batch_2/model-064.hdf5'),
                 ]

        return tx_models, tz_models, rx_models, rz_models

    # 3D motion thin slice thickness
    def CNN_3D_motion_thin_slice(self):
        tx_models = [os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_0/model-046.hdf5'), # lowest val_tx_loss
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_2/model-020.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_3/model-041.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_0/model-038.hdf5'), 
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_2/model-050.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_3/model-020.hdf5'),
                    #  os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_0/model-049.hdf5'), 
                    #  os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_2/model-051.hdf5'),
                    #  os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_3/model-026.hdf5'),
                  ]

        tz_models = [os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_0/model-031.hdf5'), # lowest val_tx_loss
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_2/model-051.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_3/model-027.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_0/model-038.hdf5'), 
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_2/model-050.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_3/model-020.hdf5'),
                    #  os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_0/model-049.hdf5'), 
                    #  os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_2/model-051.hdf5'),
                    #  os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_3/model-026.hdf5'),
                  ]


        rx_models = [os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_0/model-048.hdf5'), # lowest val_tx_loss
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_2/model-051.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_3/model-026.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_0/model-038.hdf5'), 
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_2/model-050.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_3/model-020.hdf5'),
                    #  os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_0/model-049.hdf5'), 
                    #  os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_2/model-051.hdf5'),
                    #  os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_3/model-026.hdf5'),
                  ]
        
        rz_models = [os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_0/model-038.hdf5'), # lowest val_tx_loss
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_2/model-038.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_3/model-016.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_0/model-038.hdf5'), 
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_2/model-050.hdf5'),
                     os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_3/model-020.hdf5'),
                    #  os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_0/model-049.hdf5'), 
                    #  os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_2/model-051.hdf5'),
                    #  os.path.join(self.main, 'CNN_3D_motion_thin_slice/models/batch_3/model-026.hdf5'),
                  ]

        return tx_models, tz_models, rx_models, rz_models





    def model_index(self,batch):
        if batch == 0:
            i = [0]
        
        elif batch == 1:
            i = [1]
           
        elif batch == 2:
            i = [2]
            
        elif batch == 3:
            i = [3]
        
        elif batch == 4:
            i = [4]
            
        elif batch == 5:
            i = [0,1,2,3,4]
        
        else:
            ValueError('wrong batch num')
        return i
    




    # 2D motion
    # def CNNSTN_collection(self): 
    #     theta_models = [os.path.join(self.main, 'CNNSTN_only_par/models/batch_3/model-022.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par/models/batch_2_first/model-022.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par/models/batch_0/model-010.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_2/models/batch_0/model-021.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_2/models/batch_2/model-034.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_augmentation/models/batch_0/model-006.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_on/models/batch_3/model-008.hdf5')]

    #     tx_models = [os.path.join(self.main, 'CNNSTN_only_par/models/batch_3/model-022.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par/models/batch_2/model-003.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par/models/batch_0/model-002.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_2/models/batch_0/model-015.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_2/models/batch_2/model-020.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_augmentation/models/batch_0/model-005.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_on/models/batch_3/model-008.hdf5')]

    #     ty_models = [os.path.join(self.main, 'CNNSTN_only_par/models/batch_3/model-039.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par/models/batch_2/model-008.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par/models/batch_0/model-018.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_2/models/batch_0/model-022.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_2/models/batch_2/model-041.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_augmentation/models/batch_0/model-003.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_on/models/batch_3/model-011.hdf5')]

    #     return theta_models, tx_models, ty_models

    # def model_index(self,batch):
    #     if batch == 0:
    #         i = [2,3,5]
           
    #     elif batch == 2:
    #         i = [1,4]
            
    #     elif batch == 3:
    #         i = [0,6]
            
    #     elif batch == 5:
    #         i = [0,1,2,3,4,5,6]
        
    #     else:
    #         ValueError('wrong batch num')
    #     return i


    # def CNNSTN_slice30_collection(self):
    #     tx_models = [os.path.join(self.main, 'CNNSTN_only_par_slice30/models/batch_0/model-015.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_slice30/models/batch_1/model-036.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_slice30/models/batch_2/model-023.hdf5'),]

    #     ty_models = [os.path.join(self.main, 'CNNSTN_only_par_slice30/models/batch_0/model-032.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_slice30/models/batch_1/model-032.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_slice30/models/batch_2/model-023.hdf5'),]

    #     theta_models = [os.path.join(self.main, 'CNNSTN_only_par_slice30/models/batch_0/model-014.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_slice30/models/batch_1/model-029.hdf5'),
    #               os.path.join(self.main, 'CNNSTN_only_par_slice30/models/batch_2/model-021.hdf5'),]

    #     return theta_models, tx_models, ty_models

    # def model_index_slice30(self,batch):
    #     if batch != 5:
    #         i = [batch]
           
    #     else:
    #         i = [0,1,2]

    #     return i
