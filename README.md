# Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos

[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2103.03319)
[![PWC](https://img.shields.io/badge/PWC-report-blue)](https://paperswithcode.com/paper/learning-high-fidelity-depths-of-dressed)

This repository is the official tensorflow python implementation of "[Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos](https://arxiv.org/abs/2103.03319)" in CVPR 2021 **(Oral Presentation) (Best Paper Nominated)**.

| [**Project Page**](https://www.yasamin.page/hdnet_tiktok)  | 
| ------------- | 
| [**TikTok Dataset**](https://www.yasamin.page/hdnet_tiktok#h.jr9ifesshn7v) | 


![Teaser Image](https://github.com/yasaminjafarian/HDNet_TikTok/blob/main/figures/TikTok1.gif)

This codebase provides: 
- Inference code          
- Training code           
- Visualization code      

## Requirements
(This code is tested with tensorflow-gpu 1.14.0, Python 3.7.4, CUDA 10 (version 10.0.130) and cuDNN 7 (version 7.4.2).)
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
Then install the requirements:
```sh
pip install -r requirements.txt 
```
## Inference Demo

#### Input:
The test data dimension should be: 256x256. For any test data you should have 3 **.png** files: (For an example please take a look at the demo data in "test_data" folder.)
- **name_img.png**  : The 256x256x3 test image 
- **name_mask.png** : The 256x256 corresponding mask. You can use any off-the-shelf tools such as [removebg](https://www.remove.bg/) to remove the background and get the mask. 
- **name_dp.png**   : The 256x256x3 corresponding [DensePose](http://densepose.org/). 

#### Output:
Running the demo generates the following:
- **name.txt**  : The 256x256 predicted depth
- **name_mesh.obj** : The reconstructed mesh. You can use any off-the-shelf tools such as [MeshLab](https://www.meshlab.net/) to visualize the mesh. Visualization for demo data from different views:

![Teaser Image](https://github.com/yasaminjafarian/HDNet_TikTok/blob/main/figures/mesh2.png)
- **name_normal_1.txt, name_normal_2.txt, name_normal_3.txt**   : Three 256x256 predicted normal. If you concatenate them in the third axis it will give you the 256x256x3 normal map.
- **name_results.png**  : visualization of predicted depth heatmap and the predicted normal map. Visualization for demo data:

![Teaser Image](https://github.com/yasaminjafarian/HDNet_TikTok/blob/main/figures/0043_results.png)

#### Run the demo:
Download the weights from [here](https://drive.google.com/file/d/1UOHkmwcWpwt9r11VzOCa_CVamwHVaobV/view?usp=sharing) and extract in the main repository or run this in the main repository:
```sh
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UOHkmwcWpwt9r11VzOCa_CVamwHVaobV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UOHkmwcWpwt9r11VzOCa_CVamwHVaobV" -O model.zip && rm -rf /tmp/cookies.txt

unzip model.zip
```
Run the following python code:
```
python HDNet_Inference.py
```
From line 26 to 29 under "test path and outpath" you can choose the **input directory** (default: './test_data'), **ouput directory** (default: './test_data/infer_out') and if you want to save the **visualization** (default: True).

## More Results
![Teaser Image](https://github.com/yasaminjafarian/HDNet_TikTok/blob/main/figures/TikTok2.gif)

## Training
To train the network, go to [**training folder**](https://github.com/yasaminjafarian/HDNet_TikTok/tree/main/training) and read the [**README file**](https://github.com/yasaminjafarian/HDNet_TikTok/blob/main/training/README.md)

## MATLAB Visualization
If you want to generate visualizations similar to those on the [website](https://www.yasamin.page/hdnet_tiktok#h.8chqn9lk871f), go to [**MATLAB_Visualization folder**](https://github.com/yasaminjafarian/HDNet_TikTok/tree/main/MATLAB_Visualization) and run
```
make_video.m
```
From [lines 7 to 14](https://github.com/yasaminjafarian/HDNet_TikTok/blob/main/MATLAB_Visualization/make_video.m#L7), you can choose the test folder (default: test_data) and the image name to process (default: 0043). This will generate a video of the prediction from different views (default: "test_data/infer_out/video/0043/video.avi") This process will take around 2 minutes to generate 164 angles. 

Note that this visualization will always generate a 672 × 512 video, You may want to resize your video accordingly for your own tested data.

## Citation
If you find the code or our dataset useful in your research, please consider citing the paper.

```
@InProceedings{jafarian2021tiktok,
author={Yasamin Jafarian and Hyun Soo Park},
title = {Learning High Fidelity Depths of Dressed Humans by Watching Social Media Dance Videos},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2021}}
```
