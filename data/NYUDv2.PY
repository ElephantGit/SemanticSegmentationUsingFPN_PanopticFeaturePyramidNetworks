import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from mypath import Path
from torchvision import transforms
from data import custom_transforms as tr

class NYUDSegmentation(data.Dataset):
    NUM_CLASSES = 40

    def __init__(self, args, root=Path.db_root_dir('nyudv2'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {}

        self.images_base = os.path.join(self.root, self.split, 'image')
        self.annotations_base = os.path.join(self.root, self.split, 'gtFine')

        self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.jpg')

        #self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        #self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        self.void_classes = []
        self.valid_classes = []
        for i in range(41, 895):
            self.void_classes.append(i)
        self.void_classes.remove(255)
        for i in range(1, 41):
            self.valid_classes.append(i)
        self.valid_classes.append(255)
        self.class_names = ['wall', 'floor', 'cabinet', 'bed', 'chair',\
                            'sofa', 'table', 'door', 'window', 'bookshelf',\
                            'picture', 'counter', 'blinds', 'desk', 'shelves',\
                            'curtain', 'dresser', 'pillow', 'mirror', 'floor mat',\
                            'clothes', 'ceiling', 'books', 'refridgerator', 'television',\
                            'paper', 'towel', 'shower curtain', 'box', 'whiteboard',\
                            'person', 'nightstand', 'toilet', 'sink', 'lamp',\
                            'bathtub', 'bag', 'other structure', 'other furniture', 'other prop', 'void'
                            ]

        self.ignore_index = 0
        self.class_map = dict(zip(self.valid_classes, range(1, self.NUM_CLASSES + 1)))
        self.class_map[255] = 255

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split('.')[0].split('\\')[-1] + ".png")
        # print(lbl_path)
        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            train_set = self.transform_tr(sample)
            return train_set
        elif self.split == 'val':
            val_set = self.transform_val(sample)
            return val_set
        elif self.split == 'test':
            test_set = self.transform_ts(sample)
            return test_set

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        image_names = []
        for filename in os.listdir(rootdir):
            if filename.endswith(suffix):
                image_names.append(os.path.join(rootdir, filename))
        return image_names

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            # tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            #tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from data.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    cityscapes_train = NYUDSegmentation(args, split='train')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)
