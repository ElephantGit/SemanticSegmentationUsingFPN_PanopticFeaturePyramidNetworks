# FPN-for-Semantic-Segmentation
This repository is a semantic segmentation part implement of [Kaiming He, Panoptic Feature Pyramid Networks
](https://arxiv.org/abs/1901.02446).

## Training

### Prepare data

- default dataset is

download pytorch 1.0.0 from [pytorch.org](https://pytorch.org)

download [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) dataset or[Cityscapes](https://www.cityscapes-dataset.com/)dataset

for CamVid dataset, make directory "data\CamVid" and put "701_StillsRaw_full", "LabeledApproved_full" in "CamVid".

for Cityscapes dataset, make directory "data\Cityscapes" and put "gtFine", "test", "train", "val" in "data\Cityscapes"

run:

```
data/CityScapes_utils.py    for Cityscapes dataset;
```

```
python data/CamVid_utils.py for CamVid dataset.
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
