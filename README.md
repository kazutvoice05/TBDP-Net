# TBDP-Net
This repository contains Two-Branch Depth Prediction Network (TBDP-Net) built upon [\[Hu '18\]](https://github.com/JunjH/Revisiting_Single_Depth_Estimation).

## Dependencies
- Python 3.5.1 (Anaconda4.0.0)
- Pytorch 0.4.1
- torchvision 0.2.1
- tensorflow (for visualization with tensorboard)
- tensorboardX
- matplotlib (for visualization of predicted depth maps, input images,)
- etc

## Preparing Dataset
1. Download NYU Depth V2 dataset from \[Hu '18\]'s google drive.
https://drive.google.com/file/d/1WoOZOBpOWfmwe7bknWS5PMUCLBPFKTOw/view?usp=sharing

2. Unzip the downloaded dataset zip and put it on the root directory of your workspace.

3. modify the dataset's root path defined in train.py and test.py.

## Training
You should specify the environment variable **CUDA_VISIBLE_DEVICES**.
if you don't do that, all gpus in your computer will be used for training.
you can specify the experiment name by ```-n experiment_name```. Default name is set to ```tmp```.


```CUDA_VISIBLE_DEVICES=0,1 python -W ignore train.py -n experiment_name```