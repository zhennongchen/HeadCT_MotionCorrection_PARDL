#!/usr/bin/env python

import HeadCT_motion_correction_PAR.Defaults as Defaults
import model_param
import Generator
import HeadCT_motion_correction_PAR.Data_processing as dp
import HeadCT_motion_correction_PAR.functions_collection as ff
from HeadCT_motion_correction_PAR.Build_lists import Build_list
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


def train(val_batch,trial_name,data_sheet, epochs, load_model_file, model_type):

    # build lists
    print('Build List...')
    batch_list = [0,1,2,3,4]; batch_list.pop(val_batch)
    train_batch = batch_list
    b = Build_list.Build(data_sheet)
    _, _,_, _, _, _, _, _, _,_, y_motion_param_trn, x_par_image_trn,_ = b.__build__(batch_list = train_batch)
    _, _, _, _, _, _, _,_,_,_ ,y_motion_param_val, x_par_image_val,_ = b.__build__(batch_list = [val_batch])
    
    n = 10#np.arange(0,800,4)
    x_par_image_trn = x_par_image_trn[0:n]; y_motion_param_trn = y_motion_param_trn[0:n]
    n = 10#
    x_par_image_val = x_par_image_trn[0:n]; y_motion_param_val = y_motion_param_trn[0:n]

    print(x_par_image_trn.shape, x_par_image_val.shape, x_par_image_trn[0:4],y_motion_param_trn[0:4],x_par_image_val[0:4],y_motion_param_val[0:4])

    # create model
    print('Create Model...') 
    if model_type[0:2] == 'LS':# LSTM
      input_shape = (cg.par_num,) + cg.dim + (1,)
    elif model_type[0:2] == 'CN':  #CNN
      input_shape = cg.dim + (cg.par_num,)
    print('intput shape: ', input_shape)

    model_inputs = [Input(input_shape)]
    model_outputs=[]
    if model_type[0:2] == 'LS':
      tx, ty, theta = model_param.get_LSTM(input_shape ,dimension = 3,activate = True, batch_norm = True )(model_inputs[0])
    elif model_type[0:2] == 'CN':
      tx, ty, theta = model_param.get_CNN(nb_filters=[16,32,64,128,128])(model_inputs[0])
    model_outputs += [tx]
    model_outputs += [ty]
    model_outputs += [theta]
    model = Model(inputs = model_inputs,outputs = model_outputs)
   

    if load_model_file != None:
        print(load_model_file)
        model.load_weights(load_model_file)

    # compile model
    print('Compile Model...')
    opt = Adam(lr = 1e-4)
    weights = [1,1,1]
    model.compile(optimizer= opt, 
                  loss= ['MAE','MAE','MAE'],
                  loss_weights = weights,)
    print(weights)

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
  
    datagen = Generator.DataGenerator(x_par_image_trn,y_motion_param_trn,
                                    model_type = model_type,
                                    patient_num = x_par_image_trn.shape[0], 
                                    batch_size = cg.batch_size, 
                                    input_dimension = input_shape,
                                    output_dimension = (4,),
                                    shuffle = True,
                                    add_noise = cg.add_noise,
                                    noise_sigma= cg.noise_sigma,
                                    seed = 10)


    valgen = Generator.DataGenerator(x_par_image_val,y_motion_param_val,
                                    model_type = model_type,
                                    patient_num = x_par_image_val.shape[0], 
                                    batch_size = cg.batch_size, 
                                    input_dimension = input_shape,
                                    output_dimension = (4,),
                                    shuffle = True,
                                    add_noise = False,
                                    noise_sigma = 0,
                                    seed = 10)
    print('DIMENSION: ',cg.dim)

    model.fit_generator(generator = datagen,
                        epochs = epochs,
                        validation_data = valgen,
                        callbacks = callbacks,
                        verbose = 1,
                        )


def main(val_batch):
    """These are the main training settings. Set each before running
    this file."""
    
    trial_name = 'CNN_1'
    data_sheet = os.path.join(cg.data_dir,'Patient_list/Patient_list_train_test_PAR_downsampled_MORE_Data_dim128_slice0-15.xlsx')
    
    load_model_file =  None#os.path.join(cg.model_dir,trial_name,'models','batch_2_first/model-026.hdf5')
   
    epochs = 300

    model_type = 'CNN'

    train(val_batch,trial_name, data_sheet, epochs, load_model_file, model_type)

    

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--batch', type=int)
  args = parser.parse_args()
  
  if args.batch is not None:
    assert(0 <= args.batch < 5)

  main(args.batch)
