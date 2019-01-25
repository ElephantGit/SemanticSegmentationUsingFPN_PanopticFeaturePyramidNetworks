# FPN-for-Semantic-Segmentation
This repository is a semantic segmentation part implement of [Kaiming He, Panoptic Feature Pyramid Networks
](https://arxiv.org/abs/1901.02446).

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
python trainval.py --dataset CamVid
```

train with Cityscapes(default) dataset:
change to your own CityScapes dataset path in mypath.py, then run:

```
python trainval.py --dataset Cityscapes
```

## Test


## Acknowledgment
[FCN-pytorch](https://github.com/pochih/FCN-pytorch)

[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[pytorch-fpn](https://github.com/kuangliu/pytorch-fpn)
