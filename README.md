# FPN-for-Semantic-Segmentation
This repository is a semantic segmentation part implement of [Kaiming He, Panoptic Feature Pyramid Networks
](https://arxiv.org/abs/1901.02446).

## Training

### Prepare data

- default dataset is CamVid

download pytorch 1.0.0 from [pytorch.org](https://pytorch.org)

download [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) dataset or [Cityscapes](https://www.cityscapes-dataset.com/) dataset

for CamVid dataset, make directory "data\CamVid" and put "701_StillsRaw_full", "LabeledApproved_full" in "CamVid".

for Cityscapes dataset, make directory "data\Cityscapes" and put "gtFine", "test", "train", "val" in "data\Cityscapes"

run:
for Cityscapes dataset;
```
python data/CityScapes_utils.py    
```
for CamVid dataset.
```
python data/CamVid_utils.py     
```

### Train the network

train with CamVid(default) dataset

```
python trainval.py
```

train with Cityscapes dataset

```
python trainval.py --dataset=Cityscapes
```

## Acknowledgment
[FCN-pytorch](https://github.com/pochih/FCN-pytorch)

[pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception)

[pytorch-fpn](https://github.com/kuangliu/pytorch-fpn)
