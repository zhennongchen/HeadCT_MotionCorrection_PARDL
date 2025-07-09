import numpy as np
import glob 
import os
from PIL import Image
import math
import SimpleITK as sitk
import nibabel as nb
from dipy.align.reslice import reslice
import HeadCT_MotionCorrection_PARDL.Data_processing as dp
import CTProjector.src.ct_projector.projector.numpy as ct_projector

# function: generate angle list
def get_angles_zc(nview, total_angle,start_angle):
    return np.arange(0, nview, dtype=np.float32) * (total_angle / 180 * np.pi) / nview + (start_angle / 180 * np.pi)

# function: set window level
def set_window(image,level,width):
    if len(image.shape) == 3:
        image = image.reshape(image.shape[0],image.shape[1])
    new = np.copy(image)
    high = level + width // 2
    low = level - width // 2
    # normalize
    unit = (1-0) / (width)
    new[new>high] = high
    new[new<low] = low
    new = (new - low) * unit 
    return new

# function: get first X numbers
# if we have 1000 numbers, how to get the X number of every interval numbers?
def get_X_numbers_in_interval(total_number, start_number, end_number , interval = 100):
    n = []
    for i in range(0, total_number, interval):
      n += [i + a for a in range(start_number,end_number)]
    n = np.asarray(n)
    return n


# function: find all files under the name * in the main folder, put them into a file list
def find_all_target_files(target_file_name,main_folder):
    F = np.array([])
    for i in target_file_name:
        f = np.array(sorted(glob.glob(os.path.join(main_folder, os.path.normpath(i)))))
        F = np.concatenate((F,f))
    return F

# function: find time frame of a file
def find_timeframe(file,num_of_dots,start_signal = '/',end_signal = '.'):
    k = list(file)

    if num_of_dots == 0: 
        num = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk = k[num+1:]
    
    else:
        if num_of_dots == 1: #.png
            num1 = [i for i, e in enumerate(k) if e == end_signal][-1]
        elif num_of_dots == 2: #.nii.gz
            num1 = [i for i, e in enumerate(k) if e == end_signal][-2]
        num2 = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk=k[num2+1:num1]


    total = 0
    for i in range(0,len(kk)):
        total += int(kk[i]) * (10 ** (len(kk) - 1 -i))
    return total

# function: sort files based on their time frames
def sort_timeframe(files,num_of_dots,start_signal = '/',end_signal = '.'):
    time=[]
    time_s=[]
    
    for i in files:
        a = find_timeframe(i,num_of_dots,start_signal,end_signal)
        time.append(a)
        time_s.append(a)
    time_s.sort()
    new_files=[]
    for i in range(0,len(time_s)):
        j = time.index(time_s[i])
        new_files.append(files[j])
    new_files = np.asarray(new_files)
    return new_files

# function: make folders
def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)

# function: write txt file
def txt_writer(save_path, replace , parameters,names):
    if replace == True:
        t_file = open(save_path,"w+")
    else:
        t_file = open(save_path,"a")
 
    for i in range(0,len(parameters)):
        t_file.write(names[i] + ': ')
        for ii in range(0,len(parameters[i])):
            t_file.write(str(np.round(parameters[i][ii],3))+' ')
        t_file.write('\n')
    t_file.write('\n\n')
    t_file.close()


# function: save grayscale image
def save_grayscale_image(a,save_path,normalize = True, WL = 50, WW = 100):
    I = np.zeros((a.shape[0],a.shape[1],3))
    # normalize
    if normalize == True:
        a = set_window(a, WL, WW)

    for i in range(0,3):
        I[:,:,i] = a
    
    Image.fromarray((I*255).astype('uint8')).save(save_path)


# function: normalize translation control points:
def convert_translation_control_points(t, dim, from_pixel_to_1 = True):
    if from_pixel_to_1 == True: # convert to a space -1 ~ 1
        t = [tt / dim * 2 for tt in t]
    else: # backwards
        t = [tt / 2 * dim for tt in t]
    
    return np.asarray(t)

# function: random augmentation parameters
def augment_parameters_2D(dim = [256,256], percent = 0.04, frequency = 0.5):
    do_or_not = np.random.rand()
    if do_or_not > (1-frequency):
        do = 1
        augment_tx = np.random.rand() * dim[0] * percent; augment_tx = [augment_tx if np.random.rand() > 0.5 else -augment_tx for i in range(0,1)][0]
        augment_ty = np.random.rand() * dim[1] * percent; augment_ty = [augment_ty if np.random.rand() > 0.5 else -augment_ty for i in range(0,1)][0]
        augment_r = np.random.rand() * 5; augment_r = [augment_r if np.random.rand() > 0.5 else -augment_r for i in range(0,1)][0]
        return do, augment_tx, augment_ty, augment_r
    else:
        return 0,0,0,0

# function: resample nii files
def resample_nifti(nifti, 
                   order,
                   mode, #'nearest' or 'constant' or 'reflect' or 'wrap'    
                   cval,
                   in_plane_resolution_mm=1.25,
                   slice_thickness_mm=None,
                   number_of_slices=None):
    
    # sometimes dicom to nifti programs don't define affine correctly.
    resolution = np.array(nifti.header.get_zooms()[:3] + (1,))
    if (np.abs(nifti.affine)==np.identity(4)).all():
        nifti.set_sform(nifti.affine*resolution)


    data   = nifti.get_fdata().copy()
    shape  = nifti.shape[:3]
    affine = nifti.affine.copy()
    zooms  = nifti.header.get_zooms()[:3] 

    if number_of_slices is not None:
        new_zooms = (in_plane_resolution_mm,
                     in_plane_resolution_mm,
                     (zooms[2] * shape[2]) / number_of_slices)
    elif slice_thickness_mm is not None:
        new_zooms = (in_plane_resolution_mm,
                     in_plane_resolution_mm,
                     slice_thickness_mm)            
    else:
        new_zooms = (in_plane_resolution_mm,
                     in_plane_resolution_mm,
                     zooms[2])

    new_zooms = np.array(new_zooms)
    for i, (n_i, res_i, res_new_i) in enumerate(zip(shape, zooms, new_zooms)):
        n_new_i = (n_i * res_i) / res_new_i
        # to avoid rounding ambiguities
        if (n_new_i  % 1) == 0.5: 
            new_zooms[i] -= 0.001

    data_resampled, affine_resampled = reslice(data, affine, zooms, new_zooms, order=order, mode=mode , cval = cval)
    nifti_resampled = nb.Nifti1Image(data_resampled, affine_resampled)

    x=nifti_resampled.header.get_zooms()[:3]
    y=new_zooms
    if not np.allclose(x,y, rtol=1e-02):
        print('not all close: ', x,y)

    return nifti_resampled   
    
