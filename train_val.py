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
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from data.CamVid_loader import CamVidDataset
from data import make_data_loader
from mypath import Path

from utils.metrics import Evaluator
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.loss import SegmentationLosses

from model.FPN import FPN
from model.resnet import resnet

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')
    parser.add_argument('--dataset', dest='dataset',
					    help='training dataset',
					    default='Cityscapes', type=str)
    parser.add_argument('--net', dest='net',
					    help='resnet101, res152, etc',
					    default='resnet101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
					    help='starting epoch',
					    default=1, type=int)
    parser.add_argument('--epochs', dest='epochs',
					    help='number of iterations to train',
					    default=110, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
					    help='directory to save models',
					    default=None,
					    nargs=argparse.REMAINDER)
    parser.add_argument('--num_workers', dest='num_workers',
					    help='number of worker to load data',
					    default=0, type=int)
    # cuda
    parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA'
                      default=True, type=bool)
    # multiple GPUs
    parser.add_argument('--mGPUs', dest='mGPUs', type=bool,
					    help='whether use multiple GPUs',
                        default=False,)
    parser.add_argument('--gpu_ids', dest='gpu_ids',
                        help='use which gpu to train, must be a comma-separated list of integers only (defalt=0)',
                        default='0', type=str)
    # batch size
    parser.add_argument('--batch_size', dest='batch_size',
					    help='batch_size',
					    default=None, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
					    help='training optimizer',
					    default='sgd', type=str)
    parser.add_argument('--lr', dest='lr',
					    help='starting learning rate',
					    default=0.01, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='weight_decay',
                        default=1e-5, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
					    help='step to do learning rate decay, uint is epoch',
					    default=50, type=int)
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
                        default=1, type=int)

    parser.add_argument('--checkname', dest='checkname',
                        help='checkname',
                        default=None, type=str)

    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        num_data = train_size
        self.num_per_batch = int(num_data / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if num_data % batch_size:
            self.leftover = torch.randperm(self.num_per_batch*batch_size, num_data).long()
            self.leftover_flag = True
    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return num_data

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        if args.dataset == 'CamVid':
            size = 512
            train_file = os.path.join(os.getcwd() + "\\data\\CamVid", "train.csv")
            val_file = os.path.join(os.getcwd() + "\\data\\CamVid", "val.csv")
            print('=>loading datasets')
            train_data = CamVidDataset(csv_file=train_file, phase='train')
            self.train_loader = torch.utils.data.DataLoader(train_data,
                                                     batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=args.num_workers)
            val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0)
            self.val_loader = torch.utils.data.DataLoader(val_data,
                                                     batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=args.num_workers)
            self.num_class = 32
        elif args.dataset == 'Cityscapes':
            kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
            self.train_loader, self.val_loader, self.test_loader, self.num_class = make_data_loader(args, **kwargs)
        elif args.dataset == 'NYUDv2':
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.num_class = make_data_loader(args, **kwargs)

        # Define network
        if args.net == 'resnet101':
            blocks = [2,4,23,3]
            fpn = FPN(blocks, self.num_class, back_bone=args.net)

        # Define Optimizer
        self.lr = self.args.lr
        if args.optimizer == 'adam':
            self.lr = self.lr * 0.1
            optimizer = torch.optim.Adam(fpn.parameters(), lr=args.lr, momentum=0, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(fpn.parameters(), lr=args.lr, momentum=0, weight_decay=args.weight_decay)

        # Define Criterion
        if args.dataset == 'CamVid':
            self.criterion = nn.CrossEntropyLoss()
        elif args.dataset == 'Cityscapes':
            weight = None
            self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode='ce')
        elif args.dataset == 'NYUDv2':
            weight = None
            self.criterion = SegmentationLosses(weight = weight, cuda=args.cuda).build_loss(mode='ce')

        self.model = fpn
        self.optimizer = optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.num_class)

        # multiple mGPUs
        if args.mGPUs:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume:
            output_dir = os.path.join(args.save_dir, args.dataset, args.checkname)
            runs = sorted(glob.glob(os.path.join(output_dir, 'experiment_*')))
            run_id = int(runs[-1].split('_')[-1]) - 1 if runs else 0
            experiment_dir = os.path.join(output_dir, 'experiment_{}'.format(str(run_id)))
            load_name = os.path.join(experiment_dir,
                                 'checkpoint.pth.tar')
            if not os.path.isfile(load_name):
                raise RuntimeError("=> no checkpoint found at '{}'".format(load_name))
            checkpoint = torch.load(load_name)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            self.lr = checkpoint['optimizer']['param_groups'][0]['lr']
            print("=> loaded checkpoint '{}'(epoch {})".format(load_name, checkpoint['epoch']))

        self.lr_stage = [68, 93]
        self.lr_staget_ind = 0


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        # tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        if self.lr_staget_ind > 1 and epoch % (self.lr_stage[self.lr_staget_ind]) == 0:
            adjust_learning_rate(self.optimizer, self.args.lr_decay_gamma)
            self.lr *= self.args.lr_decay_gamma
            self.lr_staget_ind += 1
        for iteration, batch in enumerate(self.train_loader):
            if self.args.dataset == 'CamVid':
                image, target = batch['X'], batch['l']
            elif self.args.dataset == 'Cityscapes':
                image, target = batch['image'], batch['label']
            elif self.args.dataset == 'NYUDv2':
                image, target = batch['image'], batch['label']
            else:
                raise NotImplementedError
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.optimizer.zero_grad()
            inputs = Variable(image)
            labels = Variable(target)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.long())
            loss_val = loss.item()
            loss.backward(torch.ones_like(loss))
            # loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            # tbar.set_description('\rTrain loss:%.3f' % (train_loss / (iteration + 1)))

            if iteration % 10 == 0:
                print("Epoch[{}]({}/{}):Loss:{:.4f}, learning rate={}".format(epoch, iteration, len(self.train_loader), loss.data, self.lr))

            self.writer.add_scalar('train/total_loss_iter', loss.item(), iteration + num_img_tr * epoch)

            #if iteration % (num_img_tr // 10) == 0:
            #    global_step = iteration + num_img_tr * epoch
            #    self.summary.visualize_image(self.witer, self.args.dataset, image, target, outputs, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, iteration * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for iter, batch in enumerate(self.val_loader):
            if self.args.dataset == 'CamVid':
                image, target = batch['X'], batch['l']
            elif self.args.dataset == 'Cityscapes':
                image, target = batch['image'], batch['label']
            elif self.args.dataset == 'NYUDv2':
                image, target = batch['image'], batch['label']
            else:
                raise NotImplementedError
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f ' % (test_loss / (iter + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/FWIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, iter * self.args.batch_size + image.shape[0]))
        print("Acc:{:.5f}, Acc_class:{:.5f}, mIoU:{:.5f}, fwIoU:{:.5f}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    args = parse_args()
    if args.save_dir is None:
        args.save_dir = os.path.join(os.getcwd(), 'run')
    if args.checkname is None:
        args.checkname = 'fpn-' + str(args.net)

    if args.cuda and args.mGPUs:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of itegers only')

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.lr is None:
        lrs = {
            'cityscapes': 0.01,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    print(args)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
    trainer.writer.close()

if __name__ == '__main__':
    main()
