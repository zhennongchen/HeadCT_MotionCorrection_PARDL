#!/usr/bin/env python

import model_STN
import Generator_STN
import HeadCT_motion_correction_PAR.Data_processing as dp
import HeadCT_motion_correction_PAR.functions_collection as ff
from HeadCT_motion_correction_PAR.Build_lists import Build_list
import HeadCT_motion_correction_PAR.Defaults as Defaults
import HeadCT_motion_correction_PAR.Hyperparameters as hyper

import argparse
import os
import numpy as np
import nibabel as nb
from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2

cg = Defaults.Parameters()
tf.random.set_seed(int(np.random.rand() * 500))

def train(batch_list, val_batch,trial_name,data_sheet, epochs, load_model_file, model_type, CP_num):

    # build lists
    print('Build List...')
    batch_list.pop(val_batch)
    train_batch = batch_list
    b = Build_list.Build(data_sheet)
    _, _, _, _, start_slice_trn, end_slice_trn, _, _, _, _, y_motion_param_trn, x_par_image_trn, _  = b.__build__(batch_list = train_batch)
    _, _, _, _, start_slice_val, end_slice_val, _, _, _, _, y_motion_param_val, x_par_image_val,_ = b.__build__(batch_list = [val_batch])
  
    n = np.arange(0,x_par_image_trn.shape[0], 3)
    x_par_image_trn = x_par_image_trn[n]; y_motion_param_trn = y_motion_param_trn[n]; start_slice_trn = start_slice_trn[n]; end_slice_trn = end_slice_trn[n]

    n = np.arange(0,x_par_image_val.shape[0], 3)
    m = ff.get_X_numbers_in_interval(n.shape[0],0,50, 100)
    x_par_image_val = x_par_image_val[n[m]]; y_motion_param_val = y_motion_param_val[n[m]]; start_slice_val = start_slice_val[n[m]]; end_slice_val = end_slice_val[n[m]]

    print(x_par_image_trn.shape, x_par_image_val.shape, 
          x_par_image_trn[0:3],y_motion_param_trn[0:3],  start_slice_trn[0:3], end_slice_trn[0:3],
          x_par_image_val[0:3],y_motion_param_val[0:3],  start_slice_val[0:3], end_slice_val[0:3],)
    
    # create model
    print('Create Model...') 
    if model_type[0:2] == 'LS':# LSTM
      input_shape = (cg.par_num,) + cg.dim + (1,)
    elif model_type[0:2] == 'CN':  #CNN
      input_shape = cg.dim + (cg.par_num,)
   
    model_inputs = [Input(input_shape)]
    model_outputs=[]
    if model_type[0:2] == 'LS':
      tx, ty, theta, final_image = model_STN.get_LSTM(input_shape ,dimension = 3, before_LSTM_filter = 512, LSTM_filter =256, after_LSTM_filter = 64, crop_STN = True, crop_size = 30)(model_inputs[0])
    elif model_type[0:2] == 'CN':
      tx, ty, tz, rx, ry, rz = model_STN.get_CNN(nb_filters = [16,32,64,128,256], dimension = 3, crop_STN = True, crop_size = 30, CP_num = CP_num)(model_inputs[0])
      # tx, ty, theta = model_STN.get_CNN(nb_filters = [16,32,64,128,256], dimension = 3, crop_STN = True, crop_size = 30)(model_inputs[0])
    model_outputs = [tx, ty, tz, rx , ry, rz]
    # model_outputs = [tx, ty, theta]

    model = Model(inputs = model_inputs,outputs = model_outputs)
  
    if load_model_file != None:
        print(load_model_file)
        model.load_weights(load_model_file)

    # compile model
    print('Compile Model...')
    opt = Adam(lr = 1e-4)
    weights = [1,1,1,1,1,1]
    model.compile(optimizer= opt, 
                  loss= ['MSE','MSE','MSE', 'MSE', 'MSE', 'MSE'],
                  loss_weights = weights,)
    print('weigths: ',weights)

    # set callbacks
    print('Set callbacks...')
    model_fld = os.path.join(cg.model_dir,trial_name,'models','batch_'+str(val_batch))
    model_name = 'model' #+ trial_name
    filepath=os.path.join(model_fld,  model_name +'-{epoch:03d}.hdf5')
    print('filepath is: ',filepath)
    ff.make_folder([os.path.dirname(os.path.dirname(model_fld)), os.path.dirname(model_fld), model_fld, os.path.join(os.path.dirname(os.path.dirname(model_fld)), 'logs')])
    csv_logger = CSVLogger(os.path.join(os.path.dirname(os.path.dirname(model_fld)), 'logs',model_name + '_batch'+ str(val_batch) + '_training-log.csv')) # log will automatically record the train_accuracy/loss and validation_accuracy/loss in each epoch
    callbacks = [csv_logger,
                    ModelCheckpoint(filepath,          
                                    monitor='val_loss',
                                    save_best_only=False,),
                     LearningRateScheduler(hyper.learning_rate_step_decay_classic),   # learning decay
                    ]
    # save model summnary
    with open(os.path.join(cg.model_dir, trial_name,'model_summary.txt'), 'w') as f:
      with redirect_stdout(f):
          model.summary()

    # # Fit
    print('Fit model...')
  
    datagen = Generator_STN.DataGenerator(x_par_image_trn,
                                          y_motion_param_trn, 
                                          start_slice_trn,
                                          end_slice_trn,
                                          start_slice_sampling = np.array([5,20,35]),#np.array([5,20,35] for slice thickness = 2.5mm or np.array([60,120,180]) for slice thickness = 0.625mm when using PARs_slice_5.nii.gz; np.array([3,4,5,6,7,17,18,19,20,21,22,23, 32, 33 ,34, 35]) when using PARs_ds_crop.nii.gz for 2.5mm or np.array([1,2,3,4,5,6,7,8,9, 26,27,28,29,30,31,32,33,34, 51,52,53,54,55,56,57,58,59]) if using PARs_ds_crop_anneal.nii.gz for 0.625mm
                                          patient_num = x_par_image_trn.shape[0], 
                                          batch_size = cg.batch_size, 
                                          input_dimension = input_shape,
                                          output_vector_dimension = (CP_num - 1,),
                                          # output_img_dimension = (60,60,2),# for 2D when crop size = 30
                                          shuffle = True,
                                          augment = False,
                                          seed = 10)


    valgen = Generator_STN.DataGenerator(x_par_image_val,
                                         y_motion_param_val,
                                         start_slice_val,
                                         end_slice_val,
                                         start_slice_sampling = np.array([5,20,35]),#np.array([5,20,35]) or np.array([60,120,180])
                                         patient_num = x_par_image_val.shape[0], 
                                         batch_size = cg.batch_size, 
                                         input_dimension = input_shape,
                                         output_vector_dimension = (CP_num - 1,),
                                         # output_img_dimension = (60,60,2),# for 2D
                                         shuffle = False,
                                         augment = False,
                                         seed = 11)
    
    model.fit_generator(generator = datagen,
                        epochs = epochs,
                        validation_data = valgen,
                        callbacks = callbacks,
                        verbose = 1,
                        )


def main(val_batch):
    """These are the main training settings. Set each before running
    this file."""
    
    trial_name = 'CNN_3D_motion_6degrees_ablation_6CP'
    model_type = 'CNN'
    CP_num = 6
    data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_3D_spline_6degrees_PAR_downsampled_slice0-50_ablation_study_6CP.xlsx')
    
    load_model_file = None#os.path.join(cg.model_dir,'CNN_3D_motion_6degrees_ablation_7CP','models','batch_1/model-000.hdf5')
   
    epochs = 100

    batch_list = [0,1,2,3,4]

    train(batch_list, val_batch,trial_name, data_sheet, epochs, load_model_file, model_type, CP_num)

    

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int)
  args = parser.parse_args()
  
  if args.batch is not None:
    assert(0 <= args.batch < 5)

  main(args.batch)
