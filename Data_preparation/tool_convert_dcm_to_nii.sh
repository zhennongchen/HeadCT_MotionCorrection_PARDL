#!/usr/bin/env bash

# run in terminal of your own laptop

##############
## Settings ##
##############

set -o nounset
set -o errexit
set -o pipefail

#shopt -s globstar nullglob

###########
## Logic ##
###########

# define the folder where dcm2niix function is saved, save it in your local laptop
dcm2niix_fld="/Users/zhennongchen/Documents/GitHub/AI_reslice_orthogonal_view/dcm2niix_11-Apr-2019/"

# main_path="/mnt/mount_zc_NAS/motion_correction/data/20221017_head_ct"
main_path="/Volumes/IRB2020P002624-DATA/zhennongchen/motion_correction/data/raw_data"
# define patient lists (the directory where you save all the patient data)
# PATIENTS=(${main_path}/dicoms_deid/thin_slice/MO*/*/*)
PATIENTS=(${main_path}/dicoms_deid/thin_slice/MO101701M000006/*/*)

echo ${#PATIENTS[@]}

filename="img"

for p in ${PATIENTS[*]};
do

  #echo ${p}
  
  if [ -d ${p} ];
  then

  patient_subid=$(basename $(dirname ${p}))
  patient_id=$(basename $(dirname $(dirname ${p})))
  echo ${patient_id}/${patient_subid}
  

  output_folder=${main_path}/nii-images/thin_slice
  mkdir -p ${output_folder}/${patient_id}/
  mkdir -p ${output_folder}/${patient_id}/${patient_subid}
  mkdir -p ${output_folder}/${patient_id}/${patient_subid}/img-nii/   # this is the folder where you save your nii files

  nii_folder=${output_folder}/${patient_id}/${patient_subid}/img-nii/  # same as above

  IMGS=(${p}/*)  # Find all the images under this patient ID
  
  for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
      do

      echo ${IMGS[${i}]}
      
      if [ "$(ls -A ${IMGS[${i}]})" ]; then  # check whether the image folder is empty
        
        
        o_file=${nii_folder}${filename}.nii.gz # define the name of output nii files, the name will be "timeframe.nii.gz"
        echo ${o_file}

        if [ -f ${o_file} ];then
          echo "already done this file"
          continue

        else
        # if dcm2niix doesn't work (error = ignore image), remove -i y
        ${dcm2niix_fld}dcm2niix -i y -m y -b n -o "${nii_folder}" -f "${filename}" -9 -z y "${IMGS[${i}]}"
        fi

      else
        echo "${IMGS[${i}]} is emtpy; Skipping"
        continue
      fi
      
    done

  else
    echo "${p} missing dicom image folder"
    continue
    
  fi

done
