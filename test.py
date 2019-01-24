from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import logging
import glob
import pandas as pd
import scipy.misc

import torch

from data.CamVid_loader import CamVidDataset
from data.utils import decode_segmap, decode_seg_map_sequence
import utils.metrics

from model.FPN import FPN
from model.resnet import resnet

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')
    parser.add_argument('--dataset', dest='dataset',
					    help='training dataset',
					    default='CamVid', type=str)
    parser.add_argument('--net', dest='net',
					    help='resnet101, res152, etc',
					    default='resnet101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
					    help='starting epoch',
					    default=1, type=int)
    parser.add_argument('--epochs', dest='epochs',
					    help='number of iterations to train',
					    default=2000, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
					    help='directory to save models',
					    default="D:\\disk\\midterm\\experiment\\code\\semantic\\fpn\\fpn\\run",
					    nargs=argparse.REMAINDER)
    parser.add_argument('--num_workers', dest='num_workers',
					    help='number of worker to load data',
					    default=0, type=int)
    # cuda
    parser.add_argument('--cuda', dest='cuda',
					    help='whether use multiple GPUs',
                        default=True,
					    action='store_true')
    # batch size
    parser.add_argument('--bs', dest='batch_size',
					    help='batch_size',
					    default=3, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
					    help='training optimizer',
					    default='sgd', type=str)
    parser.add_argument('--lr', dest='lr',
					    help='starting learning rate',
					    default=0.001, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='weight_decay',
                        default=1e-5, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
					    help='step to do learning rate decay, uint is epoch',
					    default=500, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
					    help='learning rate decay ratio',
					    default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
					    help='training session',
					    default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
					    help='resume checkpoint or not',
					    default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
					    help='checksession to load model',
					    default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
					    help='checkepoch to load model',
					    default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
					    help='checkpoint to load model',
					    default=0, type=int)

    # log and display
    parser.add_argument('--use_tfboard', dest='use_tfboard',
					    help='whether use tensorflow tensorboard',
					    default=True, type=bool)

    # configure validation
    parser.add_argument('--no_val', dest='no_val',
                        help='not do validation',
                        default=False, type=bool)
    parser.add_argument('--eval_interval', dest='eval_interval',
                        help='iterval to do evaluate',
                        default=2, type=int)

    parser.add_argument('--checkname', dest='checkname',
                        help='checkname',
                        default=None, type=str)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.dataset == 'CamVid':
        num_class = 32
    elif args.dataset == 'Cityscapes':
        num_class = 19

    if args.net == 'resnet101':
        blocks = [2, 4, 23, 3]
        model = FPN(blocks, num_class, back_bone=args.net)

    evaluator = Evaluator(num_class)

    # Trained model path and name
    output_dir = os.path.join(args.save_dir, args.dataset, args.checkname)
    runs = sorted(glob.glob(os.path.join(output_dir, 'experiment_*')))
    run_id = int(runs[-1].split('_')[-1]) - 1 if runs else 0
    experiment_dir = os.path.join(output_dir, 'experiment_{}'.format(str(run_id)))
    load_name = os.path.join(experiment_dir, 'checkpoint.pth.tar')

    # Load trained model
    if not os.path.isfile(load_name):
        raise RuntimeError("=> no checkpoint found at '{}'".format(load_name))
    checkpoint = torch.load(load_name)
    checkepoch = checkpoint['epoch']
    if args.cuda:
        self.model.load_state_dict(checkpoint['state_dict'])
    else:
        self.model.load_state_dict(checkpoint['state_dict'])

    # Load image and save in test_imgs
    test_imgs = []
    test_label = []
    if args.dataset == "CamVid":
        root_dir = "D:\\disk\\midterm\\experiment\\code\\semantic\\fpn\\fpn\\data\\CamVid"
        csv_file = os.path.join(root_dir, "val.csv")
        csv_data = pd.read_csv(csv_file)
        for i in range(len(csv_data)):
            test_imgs.append(csv_data.ix[i, 0])
            test_label.append(csv_data.ix[i, 1])

    elif args.dataset == "Cityscapes":
        root_dir = "D:\\disk\\midterm\\experiment\\code\\semantic\\fpn\\fpn\\data\\Cityscapes"
        test_dir = os.path.join(root_dir, "test")
        for scene in os.listdir(test_dir):
            dir = os.path.join(test_dir, scene)
            for img in os.listdir(dir):
                img_name = os.path.join(dir, img)
                test_imgs.append(img_name)
        test_label_dir = os.paht.join(root_dir, "gtFine", "test")
        for scene in os.listdir(test_label_dir):
            dir = os.path.join(test_label_dir, scene)
            for label in os.listdir(dir):
                label_name = os.path.join(dir, label)
                test_label.append(label_name)
    else:
        raise RuntimeError("dataset {} not found.".format(args.dataset))

    # test
    Acc = []
    Acc_class = []
    mIoU = []
    FWIoU = []
    for i in range(len(test_imgs):
        img_name = test_imgs[i]
        lebel_name = test_label[i]
        img = scipy.misc.imread(img_name, mode='RGB')
        img = torch.from_numpy(img.copy()).float()
        label = np.load(label_name)
        label = torch.from_numpy(label.copy()).long()
        img, label = img.cuda(), label.cuda()
        with torch.no_grad():
            output = model(img)
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        label = label.cpu().numpy()
        evaluator.add_batch(label, pred)

        # calculate
        Acc.append(evaluator.Pixel_Accuracy())
        Acc_class.append(evaluator.Pixel_Accuracy_Class())
        mIoU.append(evaluator.Mean_Intersection_over_Union())
        FWIoU.append(evaluator.Frequency_Weighted_Intersection_over_Union())

        # show result
        pred_rgb = decode_segmap(pred, args.dataset, plot=True)

    Acc_mean = np.array(Acc).mean()
    Acc_class_mean = np.array(Acc_class).mean()
    mIoU_mean = np.array(mIoU).mean()
    FWIoU_mean = np.array(FWIoU).mean()

    print('Mean evaluate result on dataset {}'.format(args.dataset))
    print('Acc_mean:{:.3f}\tAcc_class_mean:{:.3f}\tmIoU_mean:{:.3f}\tFWIoU_mean:{:.3f}')

if __name__ == "__main__":
    main()
