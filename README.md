# FPN-for-Semantic-Segmentation
This repository is a semantic segmentation part implement of [Kaiming He, Panoptic Feature Pyramid Networks
](https://arxiv.org/abs/1901.02446).

## To do
- [x] Semantic Segmentation Branch
  - [x] Train on CamVid Dataset
  - [ ] Train on Cityscapes Dataset
  - [ ] Train on COCO Dataset
  - [ ] Train on NYUD v2 Dataset

## results
Not a good results as the paper, I am tring to improve it.

| Dataset | mIoU | Pixel Acc | Backbone | Trained model
|-------|-------|-------|-------|-------|
| CamVid | 0.570 | 0.920 | ResNet101 | [CamVid](https://drive.google.com/file/d/1l7y6uKXhogECZd3Pw4BMl3R5TUvAA4Vw/view?usp=sharing)

## Training

### Prepare data

- default dataset is CamVid

download pytorch 1.0.0 from [pytorch.org](https://pytorch.org)

download [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) dataset or [Cityscapes](https://www.cityscapes-dataset.com/) dataset

- for CamVid dataset, make directory "data\CamVid" and put "701_StillsRaw_full", "LabeledApproved_full" in "CamVid".

- for Cityscapes dataset, make directory "Cityscapes" and put "gtFine" in "Cityscapes/gtFine_trainvaltest" folder, put "test", "train", "val" in "Cityscapes/leftImg8bit" foloder, then run:
```
python data/CityScapes_utils.py    
```

### Train the network

train with CamVid dataset:

change to your own CamVid dataset path in mypath.py, then run:

```
python train_val.py --dataset CamVid
```

train with Cityscapes(default) dataset:
change to your own CityScapes dataset path in mypath.py, then run:

```
python train_val.py --dataset Cityscapes
```

## Test
Test with CamVid dataset(val), run:
```
python test --dataset CamVid --save_dir /path/to/run
```
Test with Cityscapes dataset(val), run:
```
python test.py --dataset Cityscapes --save_dir /path/to/run
```
If you want to plot the color semantic segmentation prediction of the test input color image, please set --plot=True, for example:
```
python test.py --dataset Cityscapes --save_dir /path/to/run --plot True
```

## Acknowledgment
[FCN-pytorch](https://github.com/pochih/FCN-pytorch)

[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[pytorch-fpn](https://github.com/kuangliu/pytorch-fpn)
