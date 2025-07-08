# System
import os

class Parameters():

  def __init__(self):
  
    # # Number of partitions in the crossvalidation.
    # self.num_partitions = int(os.environ['CG_NUM_PARTITIONS'])
    
    # Dimension of padded input, for training.
    self.dim = (int(os.environ['CG_CROP_X']), int(os.environ['CG_CROP_Y']), int(os.environ['CG_CROP_Z']))
    self.par_num = int(os.environ['CG_PAR_K']) * 2 + 1
    # self.slice_num = int(os.environ['CG_CROP_Z'])

    self.unetdim = len(self.dim)
  
    # Seed for randomization.
    self.seed = int(os.environ['CG_SEED'])
      
  
    # How many images should be processed in each batch?
    self.batch_size = int(os.environ['CG_BATCH_SIZE'])

  
    # # Translation Range
    # self.xy_range = float(os.environ['CG_XY_RANGE'])
  
    # # Scale Range
    # self.zm_range = float(os.environ['CG_ZM_RANGE'])

    # # Rotation Range
    # self.rt_range=float(os.environ['CG_RT_RANGE'])
  
    # Should Flip
    self.flip = False

    # Total number of epochs to train
    self.epochs = int(os.environ['CG_EPOCHS'])
    self.lr_epochs = int(os.environ['CG_LR_EPOCHS'])
    self.start_epoch = int(os.environ['CG_START_EPOCH'])
    self.decay_rate = float(os.environ['CG_DECAY_RATE'])
    self.initial_power = float(os.environ['CG_INITIAL_POWER'])

    # noise
    if int(os.environ['CG_ADD_NOISE']) == 1:
      self.add_noise = True
    else:
      self.add_noise = False
    self.noise_sigma = float(os.environ['CG_NOISE_SIGMA'])


    # # folders
    # for VR dataset
    self.nas_dir = os.environ['CG_NAS_DIR']
    self.data_dir = os.environ['CG_DATA_DIR']
    self.model_dir = os.environ['CG_MODEL_DIR']
    self.predict_dir = os.environ['CG_PREDICT_DIR']
    self.picture_dir = os.environ['CG_PICTURE_DIR']