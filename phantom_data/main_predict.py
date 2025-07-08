#!/usr/bin/env python
import HeadCT_motion_correction_PAR.STN.model_STN as model_STN
import HeadCT_motion_correction_PAR.STN.Generator_STN as Generator_STN
import HeadCT_motion_correction_PAR.Data_processing as dp
import HeadCT_motion_correction_PAR.functions_collection as ff
from HeadCT_motion_correction_PAR.Build_lists import Build_list
import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.Trained_models.trained_models as trained_models

import os
import numpy as np
import nibabel as nb
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

cg = Defaults.Parameters()
mm = trained_models.trained_models()
model_type = 'CNN'
trial_name = 'phantom_data_6degrees'
data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_phantom_data.xlsx')
save_folder = os.path.join(cg.predict_dir,trial_name,'images')
ff.make_folder([os.path.dirname(save_folder), save_folder])

# build lists
print('Build List...')
b = Build_list.Build(data_sheet)
batches = [5]
batch_list, patient_id_list, patient_subid_list, random_name_list, start_slice_list, end_slice_list, _, _, _, _, y_motion_param_predict, x_par_image_predict, _ = b.__build__(batch_list = batches)

n = np.arange(1,10,1)
x_par_image_predict = x_par_image_predict[n]
y_motion_param_predict = y_motion_param_predict[n]
start_slice_list = start_slice_list[n]
end_slice_list = end_slice_list[n]
print(x_par_image_predict.shape, x_par_image_predict[0:2],y_motion_param_predict.shape,y_motion_param_predict[0:2])

# create model architecture:
input_shape = cg.dim + (cg.par_num,)
model_inputs = [Input(input_shape)]
model_outputs=[]
tx,ty, tz, rx,ry, rz = model_STN.get_CNN(nb_filters = [16,32,64,128,256], dimension = 3, crop_STN = True, crop_size = 30)(model_inputs[0])
model_outputs = [tx, ty, tz, rx , ry, rz]

# define model list:
model_files = mm.CNN_3D_motion_6degrees()
files = [model_files]

# do prediction
for jj in range(0,17):

  for j in range(0,len(files)): 
    f = files[j][jj]
    print(jj, j)

    model= Model(inputs = model_inputs,outputs = model_outputs)
    model.load_weights(f)
    
    for i in range(0,x_par_image_predict.shape[0]):
      patient_id = patient_id_list[n[i]]
     
      batch = batch_list[n[i]]
      
      # real motion?
      print(x_par_image_predict[i])
      if os.path.isfile(x_par_image_predict[i]) == 0:
        print('no file'); continue

      print(batch, patient_id)

      save_sub = os.path.join(save_folder, patient_id, 'parameters' )
      ff.make_folder([os.path.dirname(save_sub), save_sub])

      filename = 'model_' + str(jj) + '.npy'

      # done?
      # if os.path.isfile(os.path.join(save_sub,filename)) == 1:
      #   print('already done'); continue

      datagen = Generator_STN.DataGenerator(np.asarray([x_par_image_predict[i]]),np.asarray([y_motion_param_predict[i]]),np.asarray([start_slice_list[i]]),
                                            np.asarray([end_slice_list[i]]),
                                            start_slice_sampling = None,
                                            patient_num = 1, batch_size = cg.batch_size, 
                                            input_dimension = input_shape,output_vector_dimension = (4,),
                                            shuffle = False,augment = False,)

      tx, ty, tz, rx, ry, rz = model.predict_generator(datagen, verbose = 1, steps = 1,)
      tx = np.reshape(np.asarray(tx),-1) *5
      ty = np.reshape(np.asarray(ty),-1) *5
      tz = np.reshape(np.asarray(tz), -1) *2*2.828  #*2
      rx = np.reshape(np.asarray(rx),-1)  *5
      ry = np.reshape(np.asarray(ry),-1)  *5
      rz = np.reshape(np.asarray(rz),-1)  *5
      predict = np.reshape(np.concatenate([tx, ty, tz, rx,ry, rz], axis = -1), -1)

      np.save(os.path.join(save_sub,filename), np.reshape(predict,(6,-1)))