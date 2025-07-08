## to run this in terminal, type:
# chmod +x set_defaults.sh
# . ./set_defaults.sh   

## parameters
# define GPU you use
export CUDA_VISIBLE_DEVICES="0"

# set random seed
export CG_SEED=8

# volume dimension
export CG_CROP_X=128
export CG_CROP_Y=128
export CG_CROP_Z=15 # 15 for slicethickness = 2.5, 15 for thickness = 0.625mm
export CG_PAR_K=12

# set the batch:
export CG_BATCH_SIZE=1

# set learning epochs
export CG_EPOCHS=200
export CG_LR_EPOCHS=50 # the number of epochs for learning rate change 
export CG_START_EPOCH=0
export CG_DECAY_RATE=0.01 #0.1 if start from 0.001, 0.01 if 0.0001
export CG_INITIAL_POWER=-4

# set noise
export CG_ADD_NOISE=0
export CG_NOISE_SIGMA=8

########## need to define it if need augmentation
# # set data augmentation range
# export CG_XY_RANGE="0.1"   #0.1
# export CG_ZM_RANGE="0.1"  #0.1
# export CG_RT_RANGE="10"   #15


# folders for Zhennong's dataset (change based on your folder paths)
export CG_NAS_DIR="/mnt/camca_NAS/motion_correction/"
export CG_DATA_DIR="${CG_NAS_DIR}data/"
export CG_MODEL_DIR="${CG_NAS_DIR}model/"
export CG_PREDICT_DIR="${CG_NAS_DIR}predict/"
export CG_PICTURE_DIR="${CG_NAS_DIR}pictures/"
