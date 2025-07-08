#!/usr/bin/env bash
# run in docker 

main_path="/mnt/mount_zc_NAS/motion_correction/data/phantom_data"
# main_path="/mnt/mount_zc_NAS/head_phantom_raw/processed/study_4/scan4"
# Get a list of patients.
patients=(${main_path}/study_*)

img_folder="simulated_imgs_raw"
for p in ${patients[*]};
do

# Print the current patient.
 
  patient_subid=$(basename ${p})
  patient_id=$(basename $(dirname ${p}))
  
  echo ${p}

  # assert whether nii image exists
  if ! [ -d ${p}/${img_folder} ] || ! [ "$(ls -A  ${p}/${img_folder})" ];then
    echo "no image"
    continue
  fi

  # set output folder
  o_dir=${p}/simulated_imgs_resampled
  # echo ${o_dir}
  mkdir -p ${o_dir}


  IMGS=(${p}/${img_folder}/recon_motion_ibc_idr.nii.gz)

  for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
  do
  #echo ${IMGS[${i}]}
    i_file=${IMGS[${i}]}
    echo ${i_file}
    o_file=${o_dir}/$(basename ${i_file})
 

    if [ -f ${o_file} ];then
      echo "already done this file"
      continue
    else
      c3d ${i_file} -interpolation Cubic -resample-mm 1x1x2.828mm -o ${o_file}  #x y original: 0.601/0.6016, z original: 0.707
    fi   
  done
done


