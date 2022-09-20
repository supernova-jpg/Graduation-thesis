# Illumination-aware network 

If you think this project is helpful to you, please click the star button in the upper right corner!

This is my Pytorch implementation of my graduation thesis's deep learning framework. Since the scale of this neural network is relatively small, we only use CPU instead of a CUDA device to train the model. this project is tested on Windows 10, i7-1165G7 and there are no bugs are founded. If you want to train it on GPU, please add ```cuda()```on relevant code.
If there is any bug in my project, please feel free to contact me!

## Required Environment
 - torch 1.9.0 
 - tqdm 4.64.0  
 - numpy 1.22.4
 - opencv2-python 4.5.5.64

Before running this project, please firstly download the dataset from (https://github.com/Linfeng-Tang/MSRS.git).

## How to run our code:

### 1. Unzip the dataset ```MSRS-main.zip``` to a specific folder and then rename this foler to Ori_Dataset, move this folder into our project folder.

### 2. Run Denosing.py to obliterate noise existed in the visible light image.

### 3. Run Laplacian_Decompose.py to decompose the visible and infrared image into series of Laplacain pyramids respectively.

### 4. Run Laplacain_Fusion.py to fuse the visible and infrared image based on weighted-average strategy.

### 5. Run Train_illum.py to train the illumination classification sub-network which is designed to determine whether the  image is captured in daytime or at the night.

### 6. Run Ultimate_fuse to obtain the fused image results


The structure of our original dataset should be in this way:
```shell
 Ori_Dataset/
 ├── test
 │   ├── ir
 │   ├── Segmentation_labels
 │   ├── vi
 ├── train
 │   ├── ir
 │   ├── Segmentation_labels
 │   ├── vi
```



## If this work is helpful to you, please cite it as：
```
@article{Tang2022PIAFusion,
  title={PIAFusion: A progressive infrared and visible image fusion network based on illumination aware},
  author={Tang, Linfeng and Yuan, Jiteng and Zhang, Hao and Jiang, Xingyu and Ma, Jiayi},
  journal={Information Fusion},
  volume = {83-84},
  pages = {79-92},
  year = {2022},
  issn = {1566-2535},
  publisher={Elsevier}
}
```
