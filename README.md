# Head_CT_motion_correction_CNN
**Author: Zhennong Chen, PhD**<br />

This repo builds a U-Net with ResNet blocks, taking the motion-corrupted image as the input and outputs the motion-free image (image->image). The whole pipeline (training and prediction) is written for 2D images, but the U-Net (CNN.py) can be used for both 2D and 3D inputs. 

This repo is compatible with tensorflow-gpu 2.4.1 and cuda 11.0.3.

This deep learning model architecture refers to Su et al. *A deep learning method for eliminating head motion artifacts in computed tomography*, Medical Physics, DOI: 10.1002/mp.15354. The picture of this model can be found in Figure 3 in the paper.

To run this repo:<br />
- first: . ./set_defaults.sh<br />
- second: define your own image list in Build_list.py<br />
- third: python main_train.py<br />
- fourth: python main_predict.py<br />
