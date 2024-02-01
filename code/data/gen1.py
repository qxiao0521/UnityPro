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
import sys
sys.path.append('.')
sys.path.append('D:\\PytorchPro\\SpikeFPN\\code\\data')
from prophesee import dat_events_tools, npy_events_tools
from numpy.lib import recfunctions as rfn


class Conv_Bn_LeakyReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, groups=1, bn=True, b=3):
        super(Conv_Bn_LeakyReLu, self).__init__()
        if padding == None:
            padding = kernel_size // 2
        self.layer = nn.Sequential()
        self.layer.add_module('cov', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False))
        if bn:
            self.layer.add_module('bn', nn.BatchNorm2d(out_channels))
        self.layer.add_module('spike', nn.LeakyReLU())

    def forward(self, x):
        output = self.layer(x)
        return output

class YOLOv2Tiny_ANN_Speical(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5,
                 anchor_size=None, center_sample=False, bn=True, init_channels=5, time_steps=2, spike_b=3):
        super(YOLOv2Tiny_ANN_Speical, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = [32]
        # init set
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)
        self.time_steps = time_steps

        self.conv1 = Conv_Bn_LeakyReLu(1, 32, 5, 4, 2, bn=bn, b=spike_b)
        self.conv2 = Conv_Bn_LeakyReLu(32, 64, 3, 2, 1, bn=bn, b=spike_b)
        self.conv3 = Conv_Bn_LeakyReLu(64, 128, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample1 = Conv_Bn_LeakyReLu(128, 128, 2, 2, 0, bn=False, b=spike_b)
        self.conv4 = Conv_Bn_LeakyReLu(128, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.down_sample2 = Conv_Bn_LeakyReLu(256, 256, 2, 2, 0, bn=False, b=spike_b)
        self.conv5 = Conv_Bn_LeakyReLu(256, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.down_sample3 = Conv_Bn_LeakyReLu(256, 256, 2, 1, 0, bn=False, b=spike_b)
        self.conv6 = Conv_Bn_LeakyReLu(256, 256, 3, 1, 1, bn=bn, b=spike_b)
        self.conv7 = Conv_Bn_LeakyReLu(256, 256, 3, 1, 1, bn=bn, b=spike_b)

        self.pred = nn.Conv2d(256, 32, kernel_size=1, bias=False)

    

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # generate grid cells
        ws, hs = w // self.stride[0], h // self.stride[0]
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs * ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_xy, self.anchor_wh = self.create_grid(input_size)

    def decode_xywh(self, txtytwth_pred):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                xywh_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # b_x = sigmoid(tx) + gride_x,  b_y = sigmoid(ty) + gride_y
        B, HW, ab_n, _ = txtytwth_pred.size()
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_xy
        # b_w = anchor_w * exp(tw),     b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """
            Input:
                txtytwth_pred : [B, H*W, anchor_n, 4] containing [tx, ty, tw, th]
            Output:
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] containing [xmin, ymin, xmax, ymax]
        """
        # [H*W*anchor_n, 4]
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # [center_x, center_y, w, h] -> [xmin, ymin, xmax, ymax]
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred
    
    def decode_bbox(self, reg_pred):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy) * self.stride[0]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy) * self.stride[0]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred


    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, -1, H, W)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.down_sample1(self.conv3(x2))
        x4 = self.down_sample2(self.conv4(x3))
        x5 = self.down_sample3(self.zero_pad(self.conv5(x4)))

        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        prediction = self.pred(x7)

        B, abC, H, W = prediction.size()

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W*num_anchor, 1]
        conf_pred = prediction[:, :, :1 * self.num_anchors].contiguous().view(B, H * W * self.num_anchors, 1)
        # [B, H*W*num_anchor, num_cls]
        cls_pred = prediction[:, :, 1 * self.num_anchors: (1 + self.num_classes) * self.num_anchors].contiguous().view(
            B, H * W * self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.num_anchors:(1+self.num_classes+4) * self.num_anchors].contiguous().view(B, H * W, self.num_anchors, 4)
        # [B, H*W*num_anchor, 4]
        # 该步将原本的坐标预测从相对于grid的(0,1)数据准换为相对于整个图片的(0,1)数据
        x1y1x2y2_pred = self.decode_bbox(txtytwth_pred) / self.input_size

        return conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred


class Gen1(object):
    def __init__(self, root, object_classes, height, width, augmentation=False, mode='train', 
                ms_per_frame = 10, frame_per_sequence=5, T = 5, shuffle=True, transform=None):
        """
        Creates an iterator over the Gen1 dataset.

        :param root: path to dataset root
        :param object_classes: list of string containing objects or 'all' for all classes
        :param height: height of dataset image
        :param width: width of dataset image
        :param ms_per_frame: sbt frame interval
        :param ms_per_frame: number of frame per sequence
        :param T: prev T sequence
        :param augmentation: flip, shift and random window start for training
        :param mode: 'train', 'test' or 'val'
        """

        file_dir = os.path.join('detection_dataset_duration_60s_ratio_1.0', mode)
        self.files = os.listdir(os.path.join(root, file_dir))
        # Remove duplicates (.npy and .dat)
        self.files = [os.path.join(file_dir, time_seq_name[:-9]) for time_seq_name in self.files
                      if time_seq_name[-3:] == 'npy']

        self.root = root
        self.mode = mode
        self.width = width
        self.height = height
        self.ms_per_frame = ms_per_frame
        self.frame_per_sequence = frame_per_sequence
        self.T = T
        self.augmentation = augmentation
        self.transform = transform
        self.window_time = ms_per_frame * 1000 * frame_per_sequence * T

        if object_classes == 'all':
            self.nr_classes = 2
            self.object_classes = ['car', "pedestrian"]
        else:
            self.nr_classes = len(object_classes)
            self.object_classes = object_classes

        self.sequence_start = []
        self.sequence_time_start = []
        self.labels = []
        self.createAllBBoxDataset()
        self.nr_samples = len(self.files)

        if shuffle:
            zipped_lists = list(zip(self.files,  self.sequence_start))
            # random.seed(7)
            random.shuffle(zipped_lists)
            self.files, self.sequence_start = zip(*zipped_lists)
    
    def createAllBBoxDataset(self):
        """
        Iterates over the files and stores for each unique bounding box timestep the file name and the index of the
            unique indices file.
        """
        file_name_bbox_id = []
        print('Building Gen1 Dataset:{} set'.format(self.mode))
        pbar = tqdm.tqdm(total=len(self.files), unit='File', unit_scale=True)

        for i_file, file_name in enumerate(self.files):
            bbox_file = os.path.join(self.root, file_name + '_bbox.npy')
            event_file = os.path.join(self.root, file_name + '_td.dat')
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)    #返回 start指针位置，v_type（数据名，数据类型）迭代器，ev_size每个数据item所占内存字节数，size数据item个数
            #其中，每个数据item（t,x,y,w,h,class_id,class_confidence,track_id）
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()
            labels = np.stack([dat_bbox['t'], dat_bbox['x'], dat_bbox['y'], dat_bbox['w'], dat_bbox['h'], dat_bbox['class_id']], axis=1)
            # dat_box->labels list->2D array 30*6
            if len(self.labels) == 0:
                self.labels = labels
            else:
                self.labels = np.concatenate((self.labels, labels), axis=0)

            unique_ts, unique_indices = np.unique(dat_bbox[v_type[0][0]], return_index=True)
            #找出dat_bbox中，v_type所有字段的唯一值，并返回其索引，unique_ts, unique_indices应该是两个list  跑出来确实是两个长度为20的list
            for unique_time in unique_ts:
                sequence_start_end = self.searchEventSequence(event_file, unique_time, time_before=self.window_time)
                """
                unique_time：当前迭代的一组框对应的时间戳t（像上面所说，unique_time里一共有20个这样的t）
                event_file: 当前t对应的事件流文件路径
                self.window_time：模型一次输入所对应的t数量
                返回一个二元组(start,end)表分别表示unique_time在window_time前的下标和unique_time的下标（在event_file中的索引）
                """
                self.sequence_start.append(sequence_start_end)
                # self.sequence_start里面存了所有框的（start,end）
                self.sequence_time_start.append(unique_time-self.window_time+1)

            file_name_bbox_id += [[file_name, i] for i in range(len(unique_indices))]
            #D:\PytorchPro\Data\Gen1\test_a\test_box_file_id

            #生成一个list，元素是[文件名，索引]，索引该文件下的第几个框
            pbar.update(1)

        pbar.close()
        self.files = file_name_bbox_id
        # np.save('D:\\PytorchPro\\Data\\Gen1\\test_a_NoReshape\\test_box_file_id.npy', file_name_bbox_id)
        #self.files是一个list，元素是[文件名，索引]，索引该文件下的第几个框
        # 可能需要修改，这里没有存下来，结合477行左右看一眼
    
    def __len__(self):
        return self.nr_samples


    def __getitem__(self, idx):
        """
        returns events and label, loading events from aedat
        :param idx:
        :return: x,y,t,p,  label
        """
        bbox_file = os.path.join(self.root, self.files[idx][0] + '_bbox.npy')
        event_file = os.path.join(self.root, self.files[idx][0] + '_td.dat')

        # Bounding Box
        f_bbox = open(bbox_file, "rb")
        # dat_bbox types (v_type):
        # [('ts', 'uint64'), ('x', 'float32'), ('y', 'float32'), ('w', 'float32'), ('h', 'float32'), (
        # 'class_id', 'uint8'), ('confidence', 'float32'), ('track_id', 'uint32')]
        start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
        '''
        start:文件中数据开始的位置
        v_type: 数据类型，这里是[('t', 'uint64'), ('x', 'float32'), ('y', 'float32'), ('w', 'float32'), ('h', 'float32'), ('class_id', 'uint8'), ('class_confidence', 'float32'), ('track_id', 'uint32')]
        '''
        dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
        f_bbox.close()

        unique_ts, unique_indices = np.unique(dat_bbox[v_type[0][0]], return_index=True)
        nr_unique_ts = unique_ts.shape[0]

        bbox_time_idx = self.files[idx][1]

        # Get bounding boxes at current timestep
        if bbox_time_idx == (nr_unique_ts - 1):
            end_idx = dat_bbox[v_type[0][0]].shape[0]
        else:
            end_idx = unique_indices[bbox_time_idx+1]

        bboxes = dat_bbox[unique_indices[bbox_time_idx]:end_idx]

        # Required Information ['class_id', 'x', 'y', 'w', 'h']
        np_bbox = rfn.structured_to_unstructured(bboxes)[:, [5, 1, 2, 3, 4]]
        np_bbox = self.cropToFrame(np_bbox)
        # 检查框是否在图片范围内，并进行修剪

        label = np.zeros([np_bbox.shape[0], 5])
        label[:np_bbox.shape[0], :] = np_bbox

        # Events
        t0 = time.time()
        events = self.readEventFile(event_file, self.sequence_start[idx])
        t1 = time.time()
        # print('read 1 sample need {}s'.format(t1-t0))
        frame = self.sbt_frame(events, self.sequence_time_start[idx], ms_per_frame=self.ms_per_frame, frame_per_sequence=self.frame_per_sequence, T=self.T)
        t2 = time.time()
        # print('sbt 1 sample need {}s'.format(t2-t1))
        # frame = frame.reshape(-1, self.height, self.width)
        if self.transform is not None:
            frame, label = self.transform(frame.reshape(-1, self.height, self.width), label)
        else:
            frame = frame.reshape(-1, self.height, self.width)
        t3 = time.time()
        # print('resize 1 sample need {}s'.format(t3-t2))
        h, w = frame.shape[1], frame.shape[2]
        frame = frame.reshape(self.T, self.frame_per_sequence, h, w)
        # histogram = self.generate_input_representation(events, (self.height, self.width))

        return frame.astype(np.int8), label.astype(np.int64) 
    
    def sbt_frame(self, events, start_time, ms_per_frame=10, frame_per_sequence=5, T=5):
        # time = events[:, 2]
        # events[:, 2] -= start_time
        final_frame = np.zeros((T, frame_per_sequence, self.height, self.width))
        num_events = events.shape[0]
        for i in range(num_events):
            total_index = (events[i, 2] - start_time) // (ms_per_frame * 1000)
            frame_index = int(total_index % frame_per_sequence)
            sequence_index = int(total_index // frame_per_sequence)
            # print(total_index, sequence_index, frame_index, events[i, 1], events[i, 0])
            final_frame[sequence_index, frame_index, events[i, 1], events[i, 0]] += events[i, 3]
        return np.sign(final_frame)
        # return final_frame

    
    def searchEventSequence(self, event_file, bbox_time, time_before=250000):
        """
        Code adapted from:
        https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/master/src/io/psee_loader.py

        go to the time final_time inside the file. This is implemented using a binary search algorithm
        :param final_time: expected time
        :param term_cirterion: (nb event) binary search termination criterion
        it will load those events in a buffer and do a numpy searchsorted so the result is always exact
        """
        start_time = max(0, bbox_time - time_before + 1)

        nr_events = dat_events_tools.count_events(event_file)
        # 返回这个文件中的(x,y,t,p)数量
        file_handle = open(event_file, "rb")
        ev_start, ev_type, ev_size, img_size = dat_events_tools.parse_header(file_handle)
        #返回 ev_start 数据存储起始位置, ev_type 数据类型 ,ev_size 每个数据所占字节数 ,img_size = (240,308) 图片大小
        low = 0
        high = nr_events 
        start_position = 0
        end_position = 0

        while high > low:
            middle = (low + high) // 2

            # self.seek_event(file_handle, middle)
            file_handle.seek(ev_start + middle * ev_size)   # 移动读写指针到middle
            mid = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')], count=1)["ts"][0] #读取middle的事件戳

            if mid > start_time:
                high = middle
            elif mid < start_time:
                low = middle + 1
            else:
                # file_handle.seek(ev_start + (middle - (term_criterion // 2) * ev_size))
                break
        file_handle.seek(ev_start + low * ev_size)
        buffer = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')], count=high-low)["ts"]
        final_index = np.searchsorted(buffer, start_time, side='left')
        start_position = low + final_index  #start_time在本文件中对应的下标

        low = 0
        high = nr_events 
        while high > low:
            middle = (low + high) // 2

            # self.seek_event(file_handle, middle)
            file_handle.seek(ev_start + middle * ev_size)
            mid = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')], count=1)["ts"][0]

            if mid > bbox_time:
                high = middle
            elif mid < bbox_time:
                low = middle + 1
            else:
                # file_handle.seek(ev_start + (middle - (term_criterion // 2) * ev_size))
                break
        file_handle.seek(ev_start + low * ev_size)
        buffer = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')], count=high-low)["ts"]
        final_index = np.searchsorted(buffer, bbox_time, side='right')
        end_position = low + final_index    #bbox_time在本文件中对应的下标

        # file_handle.seek(ev_start + max(0, start_position - 1) * ev_size)
        # find_time1 = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')], count=2)["ts"]
        # file_handle.seek(ev_start + (end_position-1) * ev_size)
        # find_time2 = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')], count=2)["ts"]
        # print(start_time, bbox_time)
        # print(find_time1, find_time2)
        # input()

        
        file_handle.close()
        # we now know that it is between low and high
        return start_position, end_position

    def readEventFile(self, event_file, file_position):
        file_handle = open(event_file, "rb")
        ev_start, ev_type, ev_size, img_size = dat_events_tools.parse_header(file_handle)
        # file_position = ev_start + low * ev_size
        file_handle.seek(ev_start + file_position[0] * ev_size)
        dat_event = np.fromfile(file_handle, dtype=[('ts', 'u4'), ('_', 'i4')], count=file_position[1]-file_position[0])
        file_handle.close()

        x = np.bitwise_and(dat_event["_"], 16383)
        y = np.right_shift(
            np.bitwise_and(dat_event["_"], 268419072), 14)
        p = np.right_shift(np.bitwise_and(dat_event["_"], 268435456), 28)
        p[p == 0] = -1
        # 逐位与得到(x,y,t,p)里的（x,y,p）
        events_np = np.stack([x, y, dat_event['ts'], p], axis=-1)

        return events_np

    def cropToFrame(self, np_bbox):
        """Checks if bounding boxes are inside frame. If not crop to border"""
        pt1 = [np_bbox[:, 1], np_bbox[:, 2]]    # [[x],[y]]的list，（x,y）表示框左上角的坐标
        pt2 = [np_bbox[:, 1] + np_bbox[:, 3], np_bbox[:, 2] + np_bbox[:, 4]]    # 也是[[x],[y]]的list，但是是右下角  y变大的方向是向下
        pt1[0] = np.clip(pt1[0], 0, self.width - 1)
        pt1[1] = np.clip(pt1[1], 0, self.height - 1)
        pt2[0] = np.clip(pt2[0], 0, self.width - 1)
        pt2[1] = np.clip(pt2[1], 0, self.height - 1)
        np_bbox[:, 1] = pt1[0]
        np_bbox[:, 2] = pt1[1]
        np_bbox[:, 3] = pt2[0] - pt1[0]
        np_bbox[:, 4] = pt2[1] - pt1[1]

        # array_width = np.ones_like(np_bbox[:, 0+1]) * self.width - 1
        # array_height = np.ones_like(np_bbox[:, 1+1]) * self.height - 1

        # np_bbox[:, :2+1] = np.maximum(np_bbox[:, :2+1], np.zeros_like(np_bbox[:, :2+1]))
        # np_bbox[:, 0+1] = np.minimum(np_bbox[:, 0+1], array_width)
        # np_bbox[:, 1+1] = np.minimum(np_bbox[:, 1+1], array_height)

        # np_bbox[:, 2+1] = np.minimum(np_bbox[:, 2+1], array_width - np_bbox[:, 0+1])
        # np_bbox[:, 3+1] = np.minimum(np_bbox[:, 3+1], array_height - np_bbox[:, 1+1])

        return np_bbox

class Gen1_sbt(object):
    def __init__(self, root, object_classes, height, width, mode='train', ms_per_frame = 10, frame_per_sequence=5, T = 5, transform=None, sbt_method='mid'):
        self.file_dir = os.path.join(root, 'gen1/sbt_{}ms_{}frame_{}stack_{}'.format(ms_per_frame, frame_per_sequence, T, sbt_method), mode)
        self.files = os.listdir(self.file_dir)
        self.box_file_id = np.load(os.path.join(root, '{}_box_file_id.npy'.format(mode)))
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
        file, id = self.box_file_id[idx]
        if self.transform is not None:
            resized_frame, resized_label = self.transform(frame.reshape(-1, self.height, self.width), label)
            h, w = resized_frame.shape[1], resized_frame.shape[2]
            resized_frame = resized_frame.reshape(self.T, self.frame_per_sequence, h, w)
            return resized_frame, resized_label, label, frame, file
        return frame, label, label, file

class Gen1_sbn(object):
    def __init__(self, root, object_classes, height, width, mode='train', events_per_frame = 25000, frame_per_sequence=1, T = 2, transform=None, sbn_method='mid'):
        self.file_dir = os.path.join(root, 'sbn_{}events_{}frame_{}stack_{}'.format(events_per_frame, frame_per_sequence, T, sbn_method), mode)
        self.files = os.listdir(self.file_dir)
        self.box_file_id = np.load(os.path.join(root, '{}_box_file_id.npy'.format(mode)))
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
        file, id = self.box_file_id[idx]
        if self.transform is not None:
            resized_frame, resized_label = self.transform(frame.reshape(-1, self.height, self.width), label)
            h, w = resized_frame.shape[1], resized_frame.shape[2]
            resized_frame = resized_frame.reshape(self.T, self.frame_per_sequence, h, w)
            return resized_frame, resized_label, label, frame, file
        return frame, label, label, file

class Resize_frame(object):
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, frame, label):
        # print(frame.dtype)
        # frame.astype()
        frame = frame.transpose(1, 2, 0) #维度重排序
        h_original, w_original = frame.shape[0], frame.shape[1]
        r = min(self.input_size / h_original, self.input_size / w_original)
        h_resize, w_resize = int(round(r * h_original)), int(round(r * w_original))
        # print(r, h_original, w_original, h_resize, w_resize)
        if r > 1:
            resized_frame = cv2.resize(frame, (w_resize, h_resize), interpolation = cv2.INTER_NEAREST)  #这不是一摸一样吗？
        else:
            resized_frame = cv2.resize(frame, (w_resize, h_resize), interpolation = cv2.INTER_NEAREST)  #临近插值
        h_pad, w_pad =  self.input_size - h_resize, self.input_size - w_resize
        h_pad /= 2
        w_pad /= 2
        top, bottom = int(round(h_pad - 0.1)), int(round(h_pad + 0.1))
        left, right = int(round(w_pad - 0.1)), int(round(w_pad + 0.1))
        final_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0) # 在不够维度的部分添加黑色素
        final_label = np.zeros_like(label)
        final_label[:, 0] = label[:, 0]
        final_label[:, 1:] = np.round(label[:, 1:] * r) # 对(class,x,y,w,h)中除class以外的所有值进行放缩
        final_label[:, 1] = np.round(final_label[:, 1] + w_pad)
        final_label[:, 2] = np.round(final_label[:, 2] + h_pad)
        if len(final_frame.shape) == 2:
            final_frame = np.expand_dims(final_frame, axis=-1)
        return final_frame.transpose(2, 0, 1), final_label

def dvs_detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    ori_targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
        ori_targets.append(sample[2])
    return np.stack(imgs, 0), targets, ori_targets

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed')

    # model
    parser.add_argument('-t', '--time_steps', default=1, type=int,
                        help='spike yolo time steps')
    parser.add_argument('-tf', '--time_per_frame', default=100, type=int,
                        help='sbt time per frame')
    parser.add_argument('-fs', '--frame_per_stack', default=1, type=int,
                        help='sbt frame per stack')
    parser.add_argument('-b', '--spike_b', default=3, type=int,
                        help='spike b')
    
    
    # dataset
    parser.add_argument('-root', '--data_root', default='/media/SSD5/personal/zhanghu',
                            help='dataset root')
    
    return parser.parse_args()

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
    train_dataset = Gen1('D:/PytorchPro/Data/Gen1/val_b', object_classes='all', height=240, width=304, augmentation=False, mode='val',
                ms_per_frame = 20, frame_per_sequence=3, T = 3, shuffle=False, transform=None)
    #-----------------------------transform=Resize_frame(256) 2024/1/22 ---------------------------
    save_dir = os.path.join('D:/PytorchPro/Data/Gen1/val_b_NoReshape', 'gen1/sbt_{}ms_{}frame_{}stack_before'.format(20, 3, 3))
    train_save_dir = os.path.join(save_dir, 'val')
    if not os.path.exists(train_save_dir):
        os.makedirs(train_save_dir)
    print('start save stb trainset frame and label')
    pbar = tqdm.tqdm(total=len(train_dataset), unit='File', unit_scale=True)
    for i in range(len(train_dataset)):
        frame, label = train_dataset[i]
        frame_file_path = os.path.join(train_save_dir, 'sample{}_frame.npy'.format(i))
        label_file_path = os.path.join(train_save_dir, 'sample{}_label.npy'.format(i))
        # train_save_dir + '/sample{}_frame.npy'.format(i)
        # train_save_dir + '/sample{}_label.npy'.format(i)
        np.save(frame_file_path, frame)
        np.save(label_file_path, label)
        # if i == 64:
        #     break
        pbar.update(1)
    pbar.close()
