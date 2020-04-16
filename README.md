# Semantic Segmentation using FPN

This repository is an unofficial semantic segmentation part implementation of [Kaiming He, Panoptic Feature Pyramid Networks
](https://arxiv.org/abs/1901.02446).

## To do
- [x] Semantic Segmentation Branch
  - [x] Multiple GPUs training
  - [x] Train on CamVid Dataset
  - [x] Train on Cityscapes Dataset
  - [ ] Train on NYUD v2 Dataset
  - [ ] Train on PASCAL Context Dataset

## results
Not as good as the result in the paper, I am tring to improve it.

| Dataset | mIoU | Pixel Acc | FWIoU | Backbone | Trained model
|-------|-------|-------|-------|-------|-------|
| CamVid | 0.570 | 0.920 | 0.861 | ResNet101 | [CamVid](https://drive.google.com/file/d/1l7y6uKXhogECZd3Pw4BMl3R5TUvAA4Vw/view?usp=sharing)|
| Cityscapes | 0.605 | 0.928 | 0.872 | ResNet101 | [CityScapes](https://drive.google.com/open?id=1Dw1dyKStNo65IlQM_ORlanHLE-rmJ0ak)|

## Training

### Prepare data

- default dataset is CamVid

download pytorch 1.0.0 from [pytorch.org](https://pytorch.org)

download [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) dataset or [Cityscapes](https://www.cityscapes-dataset.com/) dataset

- for CamVid dataset, make directory "data\CamVid" and put "701_StillsRaw_full", "LabeledApproved_full" in "CamVid", then run:
```
python data/CamVid_utils.py    
```

- for Cityscapes dataset, make directory "Cityscapes" and put "gtFine" in "Cityscapes/gtFine_trainvaltest" folder, put "test", "train", "val" in "Cityscapes/leftImg8bit" foloder.

### Train the network

train with CamVid dataset:

change to your own CamVid dataset path in mypath.py, then run:

```
python train_val.py --dataset CamVid --save_dir /path/to/run
```

for **multiple GPUs training**, change to your own CamVid dataset path in mypath.py, then run:
```
python train_val.py --dataset CamVid --save_dir /path/to/run --mGPUs True --gpu_ids 0,1,2
```

train with Cityscapes(default) dataset:
change to your own CityScapes dataset path in mypath.py, then run:

```
python train_val.py --dataset Cityscapes --save_dir /path/to/run
```

for **multiple GPUs training**, change to your own CityScapes dataset path in mypath.py, then run:
```
python train_val.py --dataset Cityscapes --save_dir /path/to/run --mGPUs True --gpu_ids 0,1,2
```

## Test
Test with CamVid dataset(val), run:
```
python test --dataset CamVid --exp_dir /path/to/experiment_x
```
Test with Cityscapes dataset(val), run:
```
python test.py --dataset Cityscapes --exp_dir /path/to/experiment_x
```
If you want to plot the color semantic segmentation prediction of the test input color image, please set --plot=True, for example:
```
python test.py --dataset Cityscapes --exp_dir /path/to/experiment_x --plot True
```

## Acknowledgment
[FCN-pytorch](https://github.com/pochih/FCN-pytorch)

[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[pytorch-fpn](https://github.com/kuangliu/pytorch-fpn)

[fpn.pytorch](https://github.com/jwyang/fpn.pytorch)
