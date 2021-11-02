# Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos
[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2103.03319)

This is a tensorflow implementation of the training code for "[Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos](https://arxiv.org/abs/2103.03319)" in CVPR 2021 **(Oral Presentation)**.

| [**Project Page**](https://www.yasamin.page/hdnet_tiktok)  | 
| ------------- | 
| [**TikTok Dataset**](https://www.yasamin.page/hdnet_tiktok#h.jr9ifesshn7v) | 

## Requirements
(This code is checked with tensorflow-gpu version 1.14.0, Python 3.7.4, CUDA 10 (version 10.0.130) and cuDNN 7 (version 7.4.2).)
- numpy
- imageio
- matplotlib
- scikit-image
- scipy==1.1.0
- tensorflow-gpu==1.14.0
- gast==0.2.2
- Pillow

## Installation

Run the following code to install all pip packages:
```sh
pip install -r requirements.txt 
```
In case there is a problem, you can use the following tensorflow docker container "[(**tensorflow:19.02-py3**)](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html)":
```sh
sudo docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/tensorflow:19.02-py3
```
Then you can install the requirements:
```sh
pip install -r requirements.txt 
```
## Download Training Data and Pretrained Models
This project was trained on "[RenderPeople dataset](https://renderpeople.com/)". However, as this data is commercial, we cannot share it. Instead a sample of the public data from "[Tang et al. dataset](https://github.com/sfu-gruvi-3dv/deep_human)"  is given here for a training trial. Also a small sample of TikTok data is provided to train the semi-supervised framework. The complete TikTok dataset can be downloaded from "[here](https://www.yasamin.page/hdnet_tiktok#h.jr9ifesshn7v)". 

1. Download and extract the sample training data from [here](https://drive.google.com/file/d/1uJ_yQ0XQwNhmHI_irsx8H4f6kQ-yhp5P/view?usp=sharing) in this folder or run this:
```sh
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uJ_yQ0XQwNhmHI_irsx8H4f6kQ-yhp5P' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uJ_yQ0XQwNhmHI_irsx8H4f6kQ-yhp5P" -O training_data.zip && rm -rf /tmp/cookies.txt

unzip training_data.zip
```
2. Download and extract the pretrained model from [here](https://drive.google.com/file/d/1UOHkmwcWpwt9r11VzOCa_CVamwHVaobV/view?usp=sharing) in this folder or run this:
```sh
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UOHkmwcWpwt9r11VzOCa_CVamwHVaobV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UOHkmwcWpwt9r11VzOCa_CVamwHVaobV" -O model.zip && rm -rf /tmp/cookies.txt

unzip model.zip
```
In the end you will have three folders in this directory: (1. training_code 2. training_data 3. model)

You can reorganize your own data in this manner as well and run the training. Note that your data size should be 256x256.

## Run the Training
Here we present the training code for 1. Normal Estimator, 2. Depth Estimator 3. HDNet

First, go to the training_code directory:
```
cd training_code
```

**1. Normal Estimator**

For this, you will just need to run the python code **training_NormalEstimator.py**. Note that if you would like, you can change the variables from [line 27 to 34](https://github.com/yasaminjafarian/HDNet_training_draft/blob/ea380ca1249cc5dbe8ded9a9fe6793ba98fd0086/training_code/training_NormalEstimator.py#L27).
```
python training_NormalEstimator.py
```
This Estimator is pretraining the Network with ground truth data which is here Tang et al. data.

Every 100 steps the training results will be stored in "training_progress/visualization/NormalEstimator/Tang/"

Every 10000 steps the checkpoints will be stored in "/training_progress/model/NormalEstimator/"

**2. Depth Estimator**

For this, you will just need to run the python code **training_DepthEstimator.py**. Note that if you would like, you can change the variables from [line 27 to 34](https://github.com/yasaminjafarian/HDNet_training_draft/blob/ea380ca1249cc5dbe8ded9a9fe6793ba98fd0086/training_code/training_DepthEstimator.py#L27).
```
python training_DepthEstimator.py
```
This Estimator is pretraining the Network with ground truth data which is here Tang et al. data.

Every 100 steps the training results will be stored in "training_progress/visualization/DepthEstimator/Tang/"

Every 10000 steps the checkpoints will be stored in "/training_progress/model/DepthEstimator/"

**3. HDNet**

For this, you will just need to run the python code **training_HDNet.py**. Note that if you would like, you can change the variables from [line 27 to 34](https://github.com/yasaminjafarian/HDNet_training_draft/blob/ea380ca1249cc5dbe8ded9a9fe6793ba98fd0086/training_code/training_HDNet.py#L27). Note that this can be trained with batch size 1 only.
```
python training_HDNet.py
```
HDNet is a semi-supervised framework that is trained on both labeled data (Tang et al. data) and unlabeled data (TikTok data).

You can choose to let the network either use the pretrained models or not by commenting the [lines 128 and 129](https://github.com/yasaminjafarian/HDNet_training_draft/blob/ea380ca1249cc5dbe8ded9a9fe6793ba98fd0086/training_code/training_HDNet.py#L128)

Every 100 steps the training results on Tang et al. data will be stored in "training_progress/visualization/HDNet/Tang/"

Every 100 steps the training results on TikTok data will be stored in "training_progress/visualization/HDNet/tiktok/"

Every 10000 steps the checkpoints will be stored in "/training_progress/model/HDNet/"

## Visualize the Convergence Progress
You can visualize the progress of training convergence by running the code **plot_convergence.py**
```
python plot_convergence.py
```

## Training Data Structure
1. The training data with ground truth: (e.g. Tang_data)
```
|-color              (Images)
| |- 0000000.png
| |- 0000001.png
| |- 0000002.png
| |- ...
|-color_WO_bg        (Images with white background)
| |- 0000000.png
| |- 0000001.png
| |- 0000002.png
| |- ...
|-densepose          (DensePose IUV maps)
| |- 0000000.png
| |- 0000001.png
| |- 0000002.png
| |- ...
|-depth              (Ground truth depth values (256x256))
| |- 0000000.txt
| |- 0000001.txt
| |- 0000002.txt
| |- ...
|-mask               (human segmentation)
| |- 0000000.png
| |- 0000001.png
| |- 0000002.png
| |- ...
|-normal             (Ground truth normal)
| |- 0000000_1.txt   (first channel)
| |- 0000000_2.txt   (second channel)
| |- 0000000_3.txt   (third channel)
| |- ...
```
1. The training data without ground truth: (e.g. tiktok_data)
```
|-color              (Images)
| |- 0017634.png
| |- 0017635.png
| |- 0017636.png
| |- ...
|-color_WO_bg        (Images with white background)
| |- 0017634.png
| |- 0017635.png
| |- 0017636.png
| |- ...
|-densepose          (DensePose IUV maps)
| |- 0017634.png
| |- 0017635.png
| |- 0017636.png
| |- ...
|-correspondences    (The correspondences between two frames) (see more explanation below)
| |- corrs
| | |- 0017634_0017637_i_limit.txt
| | |- 0017634_0017637_i_r1_c1_r2_c2.txt
| | |- 0017634_0017643_i_limit.txt
| | |- 0017634_0017643_i_r1_c1_r2_c2.txt
| | |- ...
| |- corr_mat.txt
|-mask               (human segmentation)
| |- 0017634.png
| |- 0017635.png
| |- 0017636.png
| |- ...
|-pred_normals       (Predicted normals)
| |- 0017634_1.txt   (first channel)
| |- 0017634_2.txt   (second channel)
| |- 0017634_3.txt   (third channel)
| |- ...
```
You need to do some pre-processing to use unlabeled data for training: 
1. **"corr_mat.txt"** is a Nx5 matrix that in each row, the first value is the frame you want to use for self supervision (frame i) and the rest of 4 values are the frame numbres that the correspondences have been computed (frame j) and stored in **corrs** folder. (For each frame we chose 4 different frame to use for our self-supervised framework.)
2. **0017634_0017637_i_r1_c1_r2_c2.txt** is an example of one of the correspondences files. Each row has 5 values representing one 2D correspondences between frame 0017634 and frame 0017637: (**1.** i (the body part number based on densepose) **2.** r1 (row number of the correspondence in frame 0017634) **3.** c1 (column number of the correspondence in frame 0017634) **4.** r2 (row number of the correspondence in frame 0017637) **5.** c2 (column number of the correspondence in frame 0017637) )
3. **0017634_0017637_i_limit.txt** is a 24 by 3 matrix that the first column represents the 24 body part numbers from DensePose, second row and third row shows what range of rows in  **0017634_0017637_i_r1_c1_r2_c2.txt** represents that body part. If second row and third row is -1, means that the corresponding body part is not visible or doesn't have any correspondences. 

## Citation
If you find the code or our dataset useful in your research, please consider citing the paper. 

```
@InProceedings{jafarian2021tiktok,
author={Yasamin Jafarian and Hyun Soo Park},
title = {Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2021}}

@misc{jafarian2021selfsupervised,
      title={Self-supervised 3D Representation Learning of Dressed Humans from Social Media Videos}, 
      author={Yasamin Jafarian and Hyun Soo Park},
      year={2021},
      eprint={2103.03319},
      archivePrefix={arXiv},
      primaryClass={cs.CV}}
```






