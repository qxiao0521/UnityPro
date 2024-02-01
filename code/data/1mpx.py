import os
import math
import random
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
import torch.nn as nn
import cv2
import time
from prophesee import dat_events_tools, npy_events_tools
from numpy.lib import recfunctions as rfn

class OneMpx_sbt(object):
    def __init__(self, root, object_classes, height, width, mode='train', ms_per_frame = 10, frame_per_sequence=5, T = 5, transform=None, sbt_method='before'):
        self.file_dir = os.path.join(root, 'sbt_{}ms_{}frame_{}stack_{}'.format(ms_per_frame, frame_per_sequence, T, sbt_method), mode)
        self.files = os.listdir(self.file_dir)
        self.root = root
        self.mode = mode
        self.width = width
        self.height = height
        self.ms_per_frame = ms_per_frame
        self.frame_per_sequence = frame_per_sequence
        self.T = T
        self.transform = transform
        self.sbt_method = sbt_method
        if object_classes == 'all':
            self.nr_classes = 2
            self.object_classes = ['car', "pedestrian"]
        else:
            self.nr_classes = len(object_classes)
            self.object_classes = object_classes
    
    def __len__(self):
        return len(self.files) // 2


    def __getitem__(self, idx):
        """
        returns frame and label, loading them from files
        :param idx:
        :return: x,y,  label
        """
        frame = np.load(os.path.join(self.file_dir, 'sample{}_frame.npy'.format(idx))).astype(np.float32)
        label = np.load(os.path.join(self.file_dir, 'sample{}_label.npy'.format(idx))).astype(np.float32)
        if self.transform is not None:
            frame, label = self.transform(frame.reshape(-1, self.height, self.width), label)
            h, w = frame.shape[1], frame.shape[2]
            frame = frame.reshape(self.T, self.frame_per_sequence, h, w)
        return frame, label

class OneMpx_sbn(object):
    def __init__(self, root, object_classes, height, width, mode='train', events_per_frame = 25000, frame_per_sequence=1, T = 2, transform=None, sbn_method='before'):
        self.file_dir = os.path.join(root, 'sbn_{}events_{}frame_{}stack_{}'.format(events_per_frame, frame_per_sequence, T, sbn_method), mode)
        self.files = os.listdir(self.file_dir)
        self.root = root
        self.mode = mode
        self.width = width
        self.height = height
        self.events_per_frame = events_per_frame
        self.frame_per_sequence = frame_per_sequence
        self.T = T
        self.transform = transform
        self.sbn_method = sbn_method
        if object_classes == 'all':
            self.nr_classes = 2
            self.object_classes = ['car', "pedestrian"]
        else:
            self.nr_classes = len(object_classes)
            self.object_classes = object_classes
    
    def __len__(self):
        return len(self.files) // 2


    def __getitem__(self, idx):
        """
        returns frame and label, loading them from files
        :param idx:
        :return: x,y,  label
        """
        frame = np.load(os.path.join(self.file_dir, 'sample{}_frame.npy'.format(idx))).astype(np.float32)
        label = np.load(os.path.join(self.file_dir, 'sample{}_label.npy'.format(idx))).astype(np.float32)
        if self.transform is not None:
            frame, resized_label = self.transform(frame.reshape(-1, self.height, self.width), label)
            h, w = frame.shape[1], frame.shape[2]
            frame = frame.reshape(self.T, self.frame_per_sequence, h, w)
        return frame, resized_label, label

class Resize_frame(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, frame, label):
        # print(frame.dtype)
        # frame.astype()
        frame = frame.transpose(1, 2, 0)
        h_original, w_original = frame.shape[0], frame.shape[1]
        r = min(self.input_size / h_original, self.input_size / w_original)
        h_resize, w_resize = int(round(r * h_original)), int(round(r * w_original))
        # print(r, h_original, w_original, h_resize, w_resize)
        if r > 1:
            resized_frame = cv2.resize(frame, (w_resize, h_resize), interpolation = cv2.INTER_NEAREST)
        else:
            resized_frame = cv2.resize(frame, (w_resize, h_resize), interpolation = cv2.INTER_NEAREST)
        h_pad, w_pad =  self.input_size - h_resize, self.input_size - w_resize
        h_pad /= 2
        w_pad /= 2
        top, bottom = int(round(h_pad - 0.1)), int(round(h_pad + 0.1))
        left, right = int(round(w_pad - 0.1)), int(round(w_pad + 0.1))
        final_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        final_label = np.zeros_like(label)
        final_label[:, 0] = label[:, 0]
        final_label[:, 1:] = np.round(label[:, 1:] * r)
        final_label[:, 1] = np.round(final_label[:, 1] + w_pad)
        final_label[:, 2] = np.round(final_label[:, 2] + h_pad)
        if len(final_frame.shape) == 2:
            final_frame = np.expand_dims(final_frame, axis=-1)
        return final_frame.transpose(2, 0, 1), final_label

if __name__ == '__main__':
    # args = parse_args()
    # if args.device != 'cpu':
    #     print('use cuda:{}'.format(args.device))
    #     # cudnn.benchmark = True
    #     # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    #     device = torch.device("cuda:{}".format(args.device))
    # else:
    #     device = torch.device("cpu")
    # train_dataset = Gen1(args.data_root, object_classes='all', height=240, width=304, augmentation=False, mode='train', 
    #             ms_per_frame = args.time_per_frame, frame_per_sequence=args.frame_per_stack, T = args.time_steps, shuffle=False, transform=Resize_frame(256))
    # train_dataloader = torch.utils.data.DataLoader(
    #                     dataset=train_dataset, 
    #                     shuffle=True,
    #                     batch_size=args.batch_size, 
    #                     collate_fn=dvs_detection_collate,
    #                     num_workers=8,
    #                     pin_memory=True
    #                     )
    # model = YOLOv2Tiny_ANN_Speical(device=device, 
    #                     input_size=256, 
    #                     num_classes=2, 
    #                     trainable=True, 
    #                     anchor_size=[[29,22], [50,35], [92,69]], 
    #                     center_sample=False,
    #                     time_steps=args.time_steps,
    #                     spike_b=args.spike_b,
    #                     init_channels=args.frame_per_stack)
    # for iter_i, (images, targets) in enumerate(train_dataloader):
    #     images = torch.tensor(images).float().to(device)
    #     conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred = model(images)
    #     print('success')
    #     input()
    # train_dataset = OneMpx_sbt('/media/SSD6/personal/zhanghu/1Mpx/SBT/', object_classes='all', height=720, width=1280, mode='train', 
    #             ms_per_frame = 8, frame_per_sequence=1, T = 2, transform=Resize_frame(640))
    # val_dataset = OneMpx_sbt('/media/SSD6/personal/zhanghu/1Mpx/SBT/', object_classes='all', height=720, width=1280, mode='val', 
    #             ms_per_frame = 8, frame_per_sequence=1, T = 2, transform=Resize_frame(640))
    test_dataset = OneMpx_sbt('/media/SSD6/personal/zhanghu/1Mpx/SBT/', object_classes='all', height=720, width=1280, mode='test', 
                ms_per_frame = 8, frame_per_sequence=1, T = 2, transform=Resize_frame(640))
    # print(len(train_dataset))
    # print(len(val_dataset))
    # print(len(test_dataset))
    # sample = train_dataset[0][0]
    # T, C, H, W = sample.shape
    # print(T, C, H, W)
    # print(np.sum(sample==1)+np.sum(sample==0)+np.sum(sample==-1), 2*640*640)
    # print(np.sum(sample[1]))
    # print(np.sum(sample!=0))
    # print(train_dataset[0][0] != 1)
    plt.ion()
    for i in range(len(test_dataset)):
        sample = test_dataset[i][0]
        T, C, H, W = sample.shape
        if (sample[0] == 0).all() or (sample[1] == 0).all():
            print(i)
            print((sample[0] == 0).all(), (sample[1] == 0).all())
            fig, ax = plt.subplots(1, T)
            for j in range(T):
                ax[j].imshow(sample[j, 0, ...], cmap='Greys_r')
            # plt.imshow(train_dataset[0][0][0, 0, ...], cmap='Greys_r')
            plt.show(block=False)
            plt.pause(2)
            plt.close(fig)
    plt.ioff()
    # save_dir = os.path.join('/media/SSD5/personal/zhanghu', 'gen1/sbt_{}ms_{}frame_{}sequence'.format(100, 1, 1))
    # train_save_dir = os.path.join(save_dir, 'train')
    # if not os.path.exists(train_save_dir):
    #     os.makedirs(train_save_dir)
    # print('start save stb trainset frame and label')
    # pbar = tqdm.tqdm(total=len(train_dataset), unit='File', unit_scale=True)
    # for i in range(len(train_dataset)):
    #     frame, label = train_dataset[i]
    #     frame_file_path = os.path.join(train_save_dir, 'sample{}_frame.npy'.format(i))
    #     label_file_path = os.path.join(train_save_dir, 'sample{}_label.npy'.format(i))
    #     # train_save_dir + '/sample{}_frame.npy'.format(i)
    #     # train_save_dir + '/sample{}_label.npy'.format(i)
    #     np.save(frame_file_path, frame)
    #     np.save(label_file_path, label)
    #     if i == 64:
    #         break
    #     pbar.update(1)
    # pbar.close()