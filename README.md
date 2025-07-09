# Correct motion artifacts in head CT using partial angle reconstruction and deep learning
**Author: Zhennong Chen, PhD**<br />

This is the GitHub repo for the published paper: <br />
*Estimate and compensate head motion in non-contrast head CT scans using partial angle reconstruction and deep learning*<br />
[paper link](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.17047)<br />
Authors: Zhennong Chen, Quanzheng Li, Dufan Wu<br />

**Citation**: Chen, Zhennong, Quanzheng Li, and Dufan Wu. "Estimate and compensate head motion in non‐contrast head CT scans using partial angle reconstruction and deep learning." Medical physics 51.5 (2024): 3309-3321.

## Description
We propose a novel approach to correct motion artifacts in head CT using partial angle reconstruction (PAR) and deep learning:<br />
(1) Partial Angle Reconstruction (PAR) is a traditional method to estimate head motion during CT acquisition. However, its performance degrades when high temporal resolution is needed, as smaller angular intervals result in severe limited-angle artifacts in PAR images.<br />
(2) Therefore, we leverage deep learning to estimate motion parameters directly from a series of PAR images.<br />
(3) Head motion is represented as 6 rigid motion parameters: (tx, ty, tz, rx, ry, rz). Each parameter is modeled using 5 control points with B-spline interpolation. Our deep learning model predicts the last 4 control points (the first is fixed at 0), producing an output matrix of shape (6, 4).<br />

## User Guideline
### Environment Setup
- You can build your own docker from the folder ```docker```, which will build a tensorflow-based docker container. <br />

### Data Preparation (we have examples available)

- if training: **NIfTI images** of head CT without motion artifacts (required for simulation) 
   - In motion simulation, we generate PAR images (model input) and record the ground truth motion parameters for supervised training.
   - You can refer to ```example_data/data/raw_data``` for original head CT scan example
   - You can refer to ```example_data/data/simulated_data_3D_spline_6degrees``` and ```/PAR_3D_spline_6degrees``` for examples of generated PAR and motion parameters.

- if prediction: **a series of PAR images** 
   - the default dimension of model input is [25,128,128,15], where 25 is the number of PAR images, [128,128] is x-y-dimension, 15 is the number of consecutive axial slices

- **A patient list** that enumerates all your cases.  
   - To understand the expected format, please refer to the file:  
     `example_data/Patient_list/patient_list.xlsx`.

- Please refer ```example_data``` folder for examples.


### Experiments
we have designed our study into 4 steps, with each step having its own jupyter notebook.<br /> 

**step1: motion simulation**: use ```step1_motion_simulation.ipynb```, it generates ground truth motion parameters and corresponding PAR images for supervised training. <br /> 

**step2: model training**: use ```step2_train.ipynb```.<br /> 

**step3: predict motion parameters**: use ```step3_predict_motion_parameters.ipynb```. The output is a (6,4) motion parameter matrix. <br /> 

**step4: use predicted motion parameters to compensate motion in reconstruction**: not included in this repo. You are encouraged to implement your own iterative reconstruction pipeline using the predicted motion parameters. Refer to Section 2.4 of the paper for recommended reconstruction strategies.<br /> 

### Additional guidelines 
Please contact chenzhennong@gmail.com for any further questions.



