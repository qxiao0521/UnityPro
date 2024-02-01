from asyncio import events
import argparse
from cProfile import label
from optparse import Values
from tkinter.messagebox import NO
import cv2
import os
import random
import time
import numpy as np
import tqdm
import tools
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('D:\\PytorchPro\\SpikeFPN\\code\\data')

from config.yolo_config import yolov2_tiny_dvs_cfg
from data.voc import VOC_CLASSES, VOCDetection
from data.coco import coco_class_index, coco_class_labels, COCODataset
from data.gen1 import Gen1, Resize_frame, Gen1_sbt, Gen1_sbn
from data.transforms import ValTransforms

from utils import create_labels
from utils.misc import TestTimeAugmentation, dvs_detection_collate, ori_target_frame_collate
from evaluator.gen1_evaluate import coco_eval

from models.yolo import build_model
from models.snndarts_retrain.new_model_2d import Cell
from models.snndarts_search.SNN import BNN_2d, RELU_2d, SNN_2d, SNN_2d_lsnn, SNN_2d_thresh

from ptflops import get_model_complexity_info
from torchinfo import summary
from utils.sort import *

parser = argparse.ArgumentParser(description='YOLO Detection')
# basic
parser.add_argument('-size', '--img_size', default=256, type=int,
                    help='img_size')
parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training')
parser.add_argument('--show', action='store_true', default=False,
                    help='show the visulization results.')
parser.add_argument('-vs', '--visual_threshold', default=0.35, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
parser.add_argument('--save_folder', default='det_results/', type=str,
                    help='Dir to save results')
# model
parser.add_argument('-m', '--model', default='yolov2_tiny_bnn',
                    help='yolov1, yolov2, yolov3, yolov3_spp, yolov3_de, '
                            'yolov4, yolo_tiny, yolo_nano')
parser.add_argument('--weight', default='D:/PytorchPro/SpikeFPN/code/yolov2_tiny_bnn_20_0.48.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.3, type=float,
                    help='NMS threshold')
parser.add_argument('--nms_thresh', default=0.5, type=float,
                    help='NMS threshold')
parser.add_argument('--center_sample', action='store_true', default=False,
                    help='center sample trick.')
parser.add_argument('-t', '--time_steps', default=3, type=int,
                    help='spike yolo time steps')
parser.add_argument('-ef', '--events_per_frame', default=25000, type=int,
                    help='spike yolo time steps')
parser.add_argument('-tf', '--time_per_frame', default=20, type=int,
                    help='spike yolo time steps')
parser.add_argument('-fs', '--frame_per_stack', default=3, type=int,
                    help='spike yolo time steps')
parser.add_argument('-b', '--spike_b', default=3, type=int,
                    help='spike b')
parser.add_argument('--bn', action='store_false', default=True, 
                    help='use bn layer')
parser.add_argument('--frame_method', default='sbt', type=str,
                        help='sbt or sbn')

# dataset
parser.add_argument('--root', default='D:/PytorchPro/Data/Gen1/test_a_NoReshapeaa',
                    help='data root')
parser.add_argument('-d', '--dataset', default='gen1',
                    help='gen1.')
parser.add_argument('--multi_anchor', action='store_true', default=False,
                        help='use multiple anchor boxes as the positive samples')
# TTA
parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                    help='use test augmentation.')

######### LEStereo params ##################
parser.add_argument('--fea_num_layers', type=int, default=10)
parser.add_argument('--fea_filter_multiplier', type=int, default=32)
parser.add_argument('--fea_block_multiplier', type=int, default=3)
parser.add_argument('--fea_step', type=int, default=3)
parser.add_argument('--net_arch_fea', default='D:/PytorchPro/Data/save/feature_network_path.npy', type=str)
parser.add_argument('--cell_arch_fea', default='D:/PytorchPro/Data/save/feature_genotype.npy', type=str)
parser.add_argument('--experiment_description', type=str, default='test',
                    help='describ the experiment')    


args = parser.parse_args()

def convert_str2index(this_str, is_b=False, is_wight=False, is_cell=False):
    if is_wight:
        this_str = this_str.split('.')[:-1] + ['conv1','weight']
    elif is_b:
        this_str = this_str.split('.')[:-1] + ['snn_optimal','b']
    elif is_cell:
        this_str = this_str.split('.')[:3]
    else:
        this_str = this_str.split('.')
    new_index = []
    for i, value in enumerate(this_str):
        if value.isnumeric():
            new_index.append('[%s]'%value)
        else:
            if i == 0:
                new_index.append(value)
            else:
                new_index.append('.'+value)
    # print(new_this_b)
    return ''.join(new_index)

def plot_sparsity():
    value_list = np.load('alif_sparsity.npy')
    print(value_list)
    fmap_list = np.array([48*128*128, 96*64*64, 96*64*64, 96*64*64, 192*32*32, 192*32*32, 192*32*32, 384*16*16, 384*16*16, 384*16*16, 768*8*8, 768*8*8, 192*32*32, 384*16*16, 768*8*8])
    print(sum(value_list*fmap_list) / sum(fmap_list))
    # input()
    name_list = ['s0', '', 'c0', '', '', 'c3', '', '', 'c6', '', '', '', 'f-s1', '', '']
    fig = plt.figure(figsize=(6.4,3.6))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    alt.ylabel('Spiking Rate', fontsize=14)
    plt.bar(range(len(value_list)), value_list, label='ALIF', tick_label=name_list)
    # plt.show()
    plt.savefig('alif_sparsity_v4.pdf')

def test():
    args = parser.parse_args()
    # device
    if args.device != 'cpu':
        print('use cuda:{}'.format(args.device))
        # cudnn.benchmark = True
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = torch.device("cuda:{}".format(args.device))
    else:
        device = torch.device("cpu")

    # seed = 123
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark = False
    # cudnn.deterministic = True

    model_name = args.model
    print('Model: ', model_name)

    # dataset and evaluator
    if args.frame_method == 'sbt':
        test_dataset = Gen1_sbt(args.root, object_classes='all', height=240, width=304, mode='test',
                ms_per_frame = args.time_per_frame, frame_per_sequence=args.frame_per_stack, T = args.time_steps, transform=Resize_frame(args.img_size), sbt_method='before')
    else:
        test_dataset = Gen1_sbn(args.root, object_classes='all', height=240, width=304, mode='test', 
                    events_per_frame = args.events_per_frame, frame_per_sequence=args.frame_per_stack, T = args.time_steps, transform=Resize_frame(args.img_size), sbn_method='before')
    test_dataloader = torch.utils.data.DataLoader(
                        dataset=test_dataset, 
                        shuffle=False,
                        batch_size=args.batch_size, 
                        collate_fn=ori_target_frame_collate,
                        num_workers=2,
                        pin_memory=True
                        )
    classes_name = test_dataset.object_classes
    num_classes = len(classes_name)
    np.random.seed(0)

    # YOLO Config
    cfg = yolov2_tiny_dvs_cfg
    #anchor在这里
    # build model
    from models.yolo.yolov2_tiny_bnn import YOLOv2Tiny_BNN as yolo_net
    # from models.yolo.yolov2_tiny_bnn import Res_SNN as yolo_net
    model = yolo_net(device=device, 
                   input_size=args.img_size, 
                   num_classes=num_classes, 
                   trainable=False, 
                   cfg=cfg, 
                   center_sample=args.center_sample,
                   time_steps=3,
                   spike_b=args.spike_b,
                   bn=args.bn,
                   init_channels=3,
                   args=args)
    anchor_size = model.anchor_list
    all_keys = [convert_str2index(name,is_cell=True) for name, value in model.named_parameters() if '_ops' in name] 
    all_keys = list(set(all_keys))
    mem_keys = list()
    for key in all_keys:
        try:
            eval('model.%s.mem'%key)
            mem_keys.append(key)
        except:
            print(key)
            pass
    print('mem',mem_keys)
    # model = build_model(args=args, 
    #                     cfg=cfg, 
    #                     device=device, 
    #                     num_classes=num_classes, 
    #                     trainable=False)

    # load weight
    # print(model)
    model.load_state_dict(torch.load(args.weight, map_location=device), strict=False)
    model = model.to(device).eval()
    model.set_mem_keys(mem_keys)
    # print('Finished loading model!')
    # np.save('alif_stem0_conv_weight_seed2023.npy', model.feature.stem0.conv1.weight.detach().cpu().numpy())
    # print(model.feature.stem0.conv1.weight.detach().cpu().numpy().shape)
    # for name, layer in model.named_modules():
    #     if isinstance(layer, SNN_2d) or isinstance(layer, SNN_2d_lsnn) or isinstance(layer, SNN_2d_thresh):
    #         layer.fuse_conv_bn()
    # print(model.feature.stem0.rho)
    # input()
    # temp_size = 256
    # flops, params = get_model_complexity_info(model.feature, (3, temp_size, temp_size), as_strings=True, print_per_layer_stat=True)
    # print('Flops:  ' + flops)
    # print('Params: ' + params)
    # summary(model.feature, input_size=(1, 3, temp_size, temp_size))
    # summary(model.head_det_3, input_size=(1, 192, temp_size//8, temp_size//8))
    # summary(model.head_det_2, input_size=(1, 384, temp_size//16, temp_size//16))
    # summary(model.head_det_1, input_size=(1, 768, temp_size//32, temp_size//32))
    # input()
    # set tracker
    # mot_tracker = Sort(max_age=1, 
    #                    min_hits=1,
    #                    iou_threshold=0.1)
    gt_label_list = []
    pred_label_list = []
    sparsity = {}
    stem0_sparsity = []
    idx2class = ['Car', "Pedestrian"]
    idx2color = ['red', 'green']
    with torch.no_grad():
        img_list = []
        for id_, data in enumerate(tqdm.tqdm(test_dataloader)):
            # print('reset all mem')
            for key in mem_keys:
                exec('model.%s.mem=None'%key)
            image, targets, original_label, original_frame, file = data

            # if id_ >= 18: break # used for generating gif file
            # print('image: ', image.shape) # (B, S, C, H, W) = (64, 3, 3, 256, 256)
            # print('targets: ', len(targets)) # B arrays (B=64)
            
            count_list = []
            area_list = []
            sparsity_list = []
            import imageio
            color = (
                (255, 0, 0),     # image value -1, blue
                (255, 255, 255), # image value  0, white
                (0, 0, 255),     # image value  1, red
                (0, 0, 0),       # image value  2, black
            )
            '''image: (B, T, C, H, W)'''
            for frame, boxes in zip(image, targets): 
                '''frame: (T, C, H, W)'''
                '''boxes: (*, 5) which is '*' of (class, x, y, w, h)'''
                frame = np.array(frame, dtype=int)
                boxes = np.array(boxes, dtype=int)
                for target in boxes:
                    x = target[1]; y = target[2]; w = target[3]; h = target[4]
                    x_ = min(x+w, frame.shape[2]-1)
                    y_ = min(y+h, frame.shape[3]-1)
                    '''sparsity statistic'''
                    # area = (x_-x)*(y_-y)
                    # area_count = np.sum(np.abs(frame[:, :, y: y_, x: x_]))
                    # area_list.append(area)
                    # count_list.append(area_count)
                    # sparsity_list.append(area_count/area)
                    '''boxes generation'''
                    # frame[:, :, y: y_, x    ] = 2
                    # frame[:, :, y: y_, x_   ] = 2
                    # frame[:, :, y,     x: x_] = 2
                    # frame[:, :, y_,    x: x_] = 2
                '''gif generation'''
                # for t, chw in enumerate(frame):
                #     for c, hw in enumerate(chw):
                #         file_name = f'lyc_2023_8_17/images/temp_{id_}_{t}_{c}.jpg'
                #         plot_data = np.array([[color[value + 1] for value in hw[row, :]] for row in range(hw.shape[0])])
                #         cv2.imwrite(file_name, plot_data)
                #         img_list.append(imageio.imread(file_name))
        # imageio.mimsave(f'../statistics/lyc20231012/test_video.gif', img_list, duration=1e-5)
        # if True:
        #     import sys; sys.exit(1)
            '''sparsity npy file generation'''
            # np.save(f'lyc_2023_8_17/data/test_area.npy', np.array(area_list))
            # np.save(f'lyc_2023_8_17/data/test_area_count.npy', np.array(count_list))
            # np.save(f'lyc_2023_8_17/data/test_sparsity.npy', np.array(sparsity_list))
            # import sys; sys.exit(1)

            for label in original_label:
                gt_label_list.append(label)
            # print(targets[0].shape, len(targets[0]))
            targets = [label.tolist() for label in targets]
            size = np.array([[
                image.shape[-1], image.shape[-2], image.shape[-1], image.shape[-2]
            ]])
            targets = create_labels.gt_creator(
                                    img_size=args.img_size, 
                                    strides=model.stride, 
                                    label_lists=targets, 
                                    anchor_size=anchor_size, 
                                    multi_anchor=args.multi_anchor,
                                    center_sample=args.center_sample)
            # to device
            image = image.float().to(device)
            # targets = targets.float().to(device)

            # forward
            conf_pred, cls_pred, reg_pred, box_pred = model(image)

            # conf_pred, cls_pred, reg_pred, box_pred = model.stream_forward(image, file)
            for name, layer in model.named_modules():
                if isinstance(layer, Cell) or isinstance(layer, SNN_2d) or isinstance(layer, SNN_2d_lsnn) or isinstance(layer, SNN_2d_thresh) or isinstance(layer, BNN_2d):
                    if 'stem0' in name:
                        for s in layer.sparsity:
                            # print(s.shape)
                            stem0_sparsity.append(s)
                    # print(name, type(layer))
                    layer.clear_sparsity()
                    # if layer.sparsity != None:
                    #     if name not in sparsity.keys():
                    #         sparsity[name] = layer.sparsity.numpy() / 3
                    #     else:
                    #         sparsity[name] = np.append(sparsity[name], layer.sparsity.numpy() / 3)
                    #     layer.clear_sparsity()
            bboxes, scores, cls_inds = tools.get_box_score(conf_pred, cls_pred, box_pred, 
                                        num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh)
            # print(len(box))
            # bboxes *= size
            bboxes = [box * size for box in bboxes]
            bboxes = [tools.resized_box_to_original(box, args.img_size, test_dataset.height, test_dataset.width) for box in bboxes]
            plt.ion()
            for i in range(len(bboxes)):
                # dets_after_tracker = mot_tracker.update(np.concatenate((bboxes[i], np.expand_dims(scores[i], 1), np.expand_dims(cls_inds[i],1)), axis=1))
                # bboxes[i], scores[i], cls_inds[i] = dets_after_tracker[:, :4], dets_after_tracker[:, 4], dets_after_tracker[:, 5]
                pred_label = []
                for j, (box, score, cls_ind) in enumerate(zip(bboxes[i], scores[i], cls_inds[i])):
                    x1 = float(box[0])
                    y1 = float(box[1])
                    x2 = float(box[2])
                    y2 = float(box[3])
                    # label = classes_name[int(cls_ind)]
                    
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    score = float(score) # object score * class score
                    A = {"image_id": id_ * args.batch_size + i, "category_id": cls_ind, "bbox": bbox,
                        "score": score} # COCO json format
                    pred_label.append(A)
                pred_label_list.append(pred_label)
                # if (id_ * args.batch_size + i) in [164, 21767, 29042]:
                #     fmap = model.feature.stem0_fmap
                #     # plt.imshow(image[i, -1, -1, ...].cpu().numpy(), cmap='gray')
                #     # ax = plt.subplot(7, 7, 1)
                #     # ax.imshow(image[i, -1, -1, ...].cpu().numpy(), cmap='gray')
                #     # # print(fmap.shape)
                #     for j in range(48):
                #         ax = plt.subplot(6, 8, j+1)
                #         ax.imshow(fmap[-1, j, ...], cmap='gray')
                #     plt.show()
                #     input()

                # if (id_ * args.batch_size + i) in [30042, 1577, 21764]:
                
                # print(id_ * args.batch_size + i)
                
                # plt.clf()
                # plt.subplot(1,2,1)
                # plt.imshow(original_frame[i, -1, ...], cmap='Greys_r')
                # for gt in gt_label_list[id_ * args.batch_size + i]:
                #     cls = gt[0]
                #     x1, y1, w, h = gt[1:]
                #     # x2, y2 = x1 + w, y1 + h
                #     lefttop_point = (int(x1), int(y1))
                #     width_height = (int(w), int(h))
                #     plt.gca().add_patch(plt.Rectangle(lefttop_point, width_height[0], width_height[1], edgecolor=idx2color[int(cls)], fill=False))
                #     plt.gca().text(lefttop_point[0], lefttop_point[1], s='%s'%(idx2class[int(cls)]), color=idx2color[int(cls)], verticalalignment='bottom')
                # plt.subplot(1,2,2)
                # plt.imshow(original_frame[i, -1, ...], cmap='Greys_r')
                # for pred in pred_label:
                #     b = pred['bbox']
                #     cls = pred['category_id']
                #     score = pred['score']
                #     lefttop_point = (int(b[0]), int(b[1]))
                #     width_height = (int(b[2]), int(b[3]))
                #     plt.gca().add_patch(plt.Rectangle(lefttop_point, width_height[0], width_height[1], edgecolor=idx2color[int(cls)], fill=False))
                #     plt.gca().text(lefttop_point[0], lefttop_point[1], s='%s %.2f'%(idx2class[int(cls)], score), color=idx2color[int(cls)], verticalalignment='bottom')
                # plt.show()
                # input()
            plt.ioff()
            # map50_95, map50 = coco_eval([gt_label_list[-1]], [pred_label], height=240, width=304, labelmap=classes_name)
            # with open('../statistics/data/mAP_record.txt', 'a') as fp:
            #     print('loader id: %d, mAP(0.5:0.95): %.7f, mAP(0.5): %.7f' %(id_, map50_95, map50), file=fp)
        # import sys; sys.exit(1)
                #     plt.savefig('sort_test_{}.pdf'.format(id_ * args.batch_size + i))
    # print(gt_label_list[0:5])
    # print(pred_label_list[0:5])
    # print('inference time(batch size = 1):{}'.format(time.time()-start_time))
    map50_95, map50 = coco_eval(gt_label_list, pred_label_list, height=240, width=304, labelmap=classes_name)
    print('test mAP(0.5:0.95):{}, mAP(0.5):{}'.format(map50_95, map50))
    # # print(len(stem0_sparsity), stem0_sparsity[0].shape)
    # # print(len(model.input_sparsity), model.input_sparsity[0].shape)
    # # print(np.mean(np.concatenate(model.input_sparsity), 0).shape)
    # print(np.mean(np.concatenate(stem0_sparsity), 0).shape)
    # # print(np.mean(np.concatenate(stem0_sparsity), 0).shape)
    # # np.save('input_sparsity.npy', np.mean(np.concatenate(model.input_sparsity), 0))
    # np.save('alif_stem0_per_channel_sparsity.npy', np.mean(np.concatenate(stem0_sparsity), 0))
    # np.save('alif_stem0_all_neuron_888.npy', np.mean(np.concatenate(stem0_sparsity), 0))
    # # print('stem0_alif_average_sparsity: {}'.format(np.mean(np.concatenate(stem0_sparsity))) )

def video_infer():
    from data.prophesee.psee_loader import PSEELoader
    args = parser.parse_args()
    # device
    if args.device != 'cpu':
        print('use cuda:{}'.format(args.device))
        # cudnn.benchmark = True
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = torch.device("cuda:{}".format(args.device))
    else:
        device = torch.device("cpu")

    model_name = args.model
    print('Model: ', model_name)

    # YOLO Config
    cfg = yolov2_tiny_dvs_cfg
    num_classes = 2
    # build model
    from models.yolo.yolov2_tiny_bnn import YOLOv2Tiny_BNN as yolo_net
    model = yolo_net(device=device, 
                   input_size=args.img_size, 
                   num_classes=num_classes, 
                   trainable=False, 
                   cfg=cfg, 
                   center_sample=args.center_sample,
                   time_steps=3,
                   spike_b=args.spike_b,
                   bn=args.bn,
                   init_channels=3,
                   args=args)
    anchor_size = model.anchor_list
    all_keys = [convert_str2index(name,is_cell=True) for name, value in model.named_parameters() if '_ops' in name] 
    all_keys = list(set(all_keys))
    mem_keys = list()
    for key in all_keys:
        try:
            eval('model.%s.mem'%key)
            mem_keys.append(key)
        except:
            print(key)
            pass
    print('mem',mem_keys)

    # load weight
    model.load_state_dict(torch.load(args.weight, map_location='cpu'), strict=False)
    model = model.to(device).eval()
    model.set_mem_keys(mem_keys)
    print('Finished loading model!')

    stem0_sparsity = []
    sparsity = {}
    td_file = "/media/HDD3/personal_files/zhanghu/detection_dataset_duration_60s_ratio_1.0/test/17-04-11_15-13-23_61500000_121500000_td.dat"
    video = PSEELoader(td_file)
    i = 0
    with torch.no_grad():
        while video.done == False:
            i += 1
            events = video.load_delta_t(50000)
            # print('{}ms'.format(i * 50))
            # image = vis.make_binary_histo(evs, img=im, width=width, height=height)
            # to device
            # image = image.float().to(device)
            frame = np.zeros((1, 240, 304))
            x, y, p = events['x'], events['y'], events['p']
            p[p==0] = -1
            # events = events[events[:, 0] < 1280]
            # total = min((events[:, 2] - start_time) // (ms_per_frame * 1000), frame_per_sequence * T-1)
            # final_frame[sequence_index, frame_index, events[:, 1], events[:, 0]] += events[:, 3]
            # num_events = events.shape[0]
            np.add.at(frame, (0, y, x), p)
            frame = np.sign(frame)
            frame = frame.transpose(1, 2, 0)
            h_original, w_original = frame.shape[0], frame.shape[1]
            r = min(args.img_size / h_original, args.img_size / w_original)
            h_resize, w_resize = int(round(r * h_original)), int(round(r * w_original))
            # print(r, h_original, w_original, h_resize, w_resize)
            if r > 1:
                resized_frame = cv2.resize(frame, (w_resize, h_resize), interpolation = cv2.INTER_NEAREST)
            else:
                resized_frame = cv2.resize(frame, (w_resize, h_resize), interpolation = cv2.INTER_NEAREST)
            h_pad, w_pad =  args.img_size - h_resize, args.img_size - w_resize
            h_pad /= 2
            w_pad /= 2
            top, bottom = int(round(h_pad - 0.1)), int(round(h_pad + 0.1))
            left, right = int(round(w_pad - 0.1)), int(round(w_pad + 0.1))
            final_frame = cv2.copyMakeBorder(resized_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            if len(final_frame.shape) == 2:
                final_frame = np.expand_dims(final_frame, axis=-1)
            final_frame = final_frame.transpose(2, 0, 1)
            final_frame = torch.from_numpy(final_frame).unsqueeze(0).unsqueeze(0).repeat(1, 3, 3, 1, 1)
            final_frame = final_frame.float().to(device)

            # forward
            conf_pred, cls_pred, reg_pred, box_pred = model.stream_forward_new(final_frame, [td_file])
            # if i in [200, 350, 420]:
            #     plt.imshow(final_frame[-1, -1, -1, ...].cpu().numpy(), cmap='gray')
            #     # fmap = model.feature.stem0_fmap
            #     # print(fmap.shape)
            #     # for j in range(48):
            #     #     ax = plt.subplot(6, 8, j+1)
            #     #     ax.imshow(fmap[-1, j, ...], cmap='gray')
            #     plt.show()
            #     input()

            for name, layer in model.named_modules():
                if isinstance(layer, Cell) or isinstance(layer, SNN_2d) or isinstance(layer, SNN_2d_lsnn) or isinstance(layer, BNN_2d) or isinstance(layer, RELU_2d):
                    if 'stem0' in name:
                        # print(np.mean(layer.sparsity))
                        for s in layer.sparsity:
                            # print(s.shape)
                            stem0_sparsity.append(s)
                        layer.clear_sparsity()
                        # stem0_sparsity.append(np.mean(layer.sparsity))
            #         # print(name, type(layer))
            #         if layer.sparsity != None:
            #             if name not in sparsity.keys():
            #                 sparsity[name] = layer.sparsity.numpy() / 3
            #             else:
            #                 sparsity[name] = np.append(sparsity[name], layer.sparsity.numpy() / 3)
            #             layer.clear_sparsity()
    # plt.plot([50 * (i+1) for i in range(len(stem0_sparsity))], stem0_sparsity)
    print(len(stem0_sparsity))
    # np.save('stem0_5x5_sparsity_alif_888.npy', stem0_sparsity)
    # plt.show()
    # plt.savefig('stem0_sparsity_relu.pdf')

def plot_density():
    fig = plt.figure(figsize=(10, 5))
    # fig, ax = plt.subplots(1, 4, figsize=(18,5))
    # lif_sparsity_stem0_123 = np.mean(np.load('lif_stem0_sparsity_123.npy'), axis=0)
    # alif_sparsity_stem0_123 = np.mean(np.load('alif_stem0_sparsity_123.npy'), axis=0)
    # lif_sparsity_stem0_2023 = np.mean(np.load('lif_stem0_sparsity_2023.npy'), axis=0)
    # alif_sparsity_stem0_2023 = np.mean(np.load('alif_stem0_sparsity_2023.npy'), axis=0)
    # lif_sparsity_stem0_888 = np.mean(np.load('lif_stem0_sparsity_888.npy'), axis=0)
    # alif_sparsity_stem0_888 = np.mean(np.load('alif_stem0_sparsity_888.npy'), axis=0)
    # lif_sparsity_stem0_999 = np.mean(np.load('lif_stem0_sparsity_999.npy'), axis=0)
    # alif_sparsity_stem0_999 = np.mean(np.load('alif_stem0_sparsity_999.npy'), axis=0)
    # bnn_sparsity_stem0 = np.mean(np.load('bnn_stem0_sparsity.npy'), axis=0)
    # ax[0].plot(list(range(len(lif_sparsity_stem0_123))), lif_sparsity_stem0_123, label='LIF', color='b')
    # ax[0].plot(list(range(len(alif_sparsity_stem0_123))), alif_sparsity_stem0_123, label='ALIF', color='g')
    # ax[0].legend(fontsize=18)
    # ax[1].plot(list(range(len(lif_sparsity_stem0_2023))), lif_sparsity_stem0_2023, label='LIF', color='b')
    # ax[1].plot(list(range(len(alif_sparsity_stem0_2023))), alif_sparsity_stem0_2023, label='ALIF', color='g')
    # ax[1].legend(fontsize=18)
    # ax[2].plot(list(range(len(lif_sparsity_stem0_888))), lif_sparsity_stem0_888, label='LIF', color='b')
    # ax[2].plot(list(range(len(alif_sparsity_stem0_888))), alif_sparsity_stem0_888, label='ALIF', color='g')
    # ax[2].legend(fontsize=18)
    # ax[3].plot(list(range(len(lif_sparsity_stem0_999))), lif_sparsity_stem0_999, label='LIF', color='b')
    # ax[3].plot(list(range(len(alif_sparsity_stem0_999))), alif_sparsity_stem0_999, label='ALIF', color='g')
    # ax[3].legend(fontsize=18)
    # print(np.expand_dims(np.mean(lif_sparsity_stem0, axis=1), 0).shape)
    # ax[0].imshow(np.expand_dims(np.mean(lif_sparsity_stem0, axis=1), 0))
    # ax[0].set_title('lif')
    # im = ax[1].imshow(np.expand_dims(np.mean(bnn_sparsity_stem0, axis=1), 0))
    # ax[1].set_title('bn')
    # lif_sparsity_stem0 = np.load('lif_stem0_per_channel_sparsity.npy').flatten()
    # alif_sparsity_stem0 = np.load('alif_stem0_per_channel_sparsity.npy').flatten()
    # bnn_sparsity_stem0 = np.load('bnn_stem0_per_channel_sparsity.npy').flatten()
    # plt.plot(list(range(len(lif_sparsity_stem0))), lif_sparsity_stem0, label='LIF', color='b')
    # plt.plot(list(range(len(alif_sparsity_stem0))), alif_sparsity_stem0, label='ALIF', color='g')
    # plt.plot(list(range(len(bnn_sparsity_stem0))), bnn_sparsity_stem0, label='BN', color='r')
    events_sparsity = np.load('events_density.npy')
    # stem0_sparsity_lif_1 = np.load('stem0_3x3_density_lif_1.npy')
    # stem0_sparsity_lif_2 = np.load('stem0_3x3_density_lif_2.npy')
    # stem0_sparsity_bnn = np.load('stem0_3x3_density_bn.npy')
    # stem0_sparsity_alif = np.load('stem0_3x3_density_alif.npy')
    # stem0_sparsity_relu = np.load('stem0_density_relu.npy')
    # time-density
    stem0_sparsity_alif = np.load('stem0_5x5_sparsity_alif_stream.npy')
    stem0_sparsity_lif_1 = np.load('stem0_5x5_sparsity_lif_stream.npy')
    stem0_sparsity_bnn = np.load('stem0_5x5_sparsity_bn_stream.npy')
    # stem0_sparsity_alif_2023 = np.load('stem0_5x5_sparsity_alif_2023.npy')
    # stem0_sparsity_lif_2023 = np.load('stem0_5x5_sparsity_lif_2023.npy')
    # stem0_sparsity_alif_888 = np.load('stem0_5x5_sparsity_alif_888.npy')
    # stem0_sparsity_lif_888 = np.load('stem0_5x5_sparsity_lif_888.npy')
    # stem0_sparsity_alif_999 = np.load('stem0_5x5_sparsity_alif_999.npy')
    # stem0_sparsity_lif_999 = np.load('stem0_5x5_sparsity_lif_999.npy')
    # ax[0].plot(list(range(250)), events_sparsity[250:500], label='Events', color='black')
    # ax[0].plot(list(range(250)), stem0_sparsity_lif[250:500], label='LIF', color='b')
    # ax[0].plot(list(range(250)), stem0_sparsity_alif[250:500], label='ALIF', color='g')
    # ax[0].plot(list(range(250)), stem0_sparsity_bnn[250:500], label='BN', color='r')
    # ax[1].plot(list(range(250)), stem0_sparsity_lif_2023[250:500], label='LIF', color='b')
    # ax[1].plot(list(range(250)), stem0_sparsity_alif_2023[250:500], label='ALIF', color='g')
    # ax[2].plot(list(range(250)), stem0_sparsity_lif_888[250:500], label='LIF', color='b')
    # ax[2].plot(list(range(250)), stem0_sparsity_alif_888[250:500], label='ALIF', color='g')
    # ax[3].plot(list(range(250)), stem0_sparsity_lif_999[250:500], label='LIF', color='b')
    # ax[3].plot(list(range(250)), stem0_sparsity_alif_999[250:500], label='ALIF', color='g')
    # plt.plot(list(range(len(events_sparsity))), events_sparsity, label='Events', color='k') # royalblue k
    # plt.plot(list(range(len(events_sparsity))), stem0_sparsity_lif_1, label='LIF', color='b') # orange b
    # # plt.plot(list(range(len(events_sparsity))), stem0_sparsity_lif_2, label='LIF-2', color='k') # k k
    # plt.plot(list(range(len(events_sparsity))), stem0_sparsity_alif, label='ALIF', color='g')   # b g
    # plt.plot(list(range(len(events_sparsity))), stem0_sparsity_bnn, label='BN', color='r') # r r
    # # plt.plot(list(range(len(events_sparsity))), stem0_sparsity_relu, label='Relu')
    '''alif_lif_bn_rate.pdf--begin'''
    plt.plot(list(range(250)), events_sparsity[250:500], label='Events', color='k') # royalblue k
    plt.plot(list(range(250)), stem0_sparsity_lif_1[250:500], label='LIF', color='b') # orange b
    plt.plot(list(range(250)), stem0_sparsity_alif[250:500], label='ALIF', color='g')   # b g
    plt.plot(list(range(250)), stem0_sparsity_bnn[250:500], label='BN', color='r') # r r
    '''alif_lif_bn_rate.pdf--end'''
    plt.legend(fontsize=18)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    # # plt.xticks(fontsize=21)
    # plt.yticks(np.arange(0, 0.20, 0.05), fontsize=21)
    # # plt.ylim([0, 0.2])
    # plt.xlabel('Column Index', fontsize=21)
    plt.xlabel('Time[60ms]', fontsize=21)
    plt.ylabel('Spiking Rate', fontsize=21)
    # plt.colorbar(im, ax=ax.ravel().tolist())
    # alif_tau_a_123 = np.load('alif_tau_a_123.npy')
    # alif_tau_a_2023 = np.load('alif_tau_a_2023.npy')
    # alif_tau_a_888 = np.load('alif_tau_a_888.npy')
    # alif_tau_a_999 = np.load('alif_tau_a_999.npy')
    # print(alif_tau_a_999)
    # ax[0].scatter(list(range(len(alif_tau_a_123))), alif_tau_a_123)
    # ax[1].scatter(list(range(len(alif_tau_a_2023))), alif_tau_a_2023)
    # ax[2].scatter(list(range(len(alif_tau_a_888))), alif_tau_a_888)
    # ax[3].scatter(list(range(len(alif_tau_a_999))), alif_tau_a_999)
    plt.tight_layout()
    # plt.show()
    # plt.savefig('stem0_5x5_sparsity_compare.pdf')
    plt.savefig('IEEE2023_figures/alif_lif_bn_rate.pdf')
    # plt.savefig('alif_lif_bn_density.pdf')

def plot_dataset_density(index_list):
    args = parser.parse_args()
    # device
    if args.device != 'cpu':
        print('use cuda:{}'.format(args.device))
        # cudnn.benchmark = True
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = torch.device("cuda:{}".format(args.device))
    else:
        device = torch.device("cpu")
    torch.backends.cudnn.benchmark = True
    model_name = args.model
    print('Model: ', model_name)

    # dataset and evaluator
    if args.frame_method == 'sbt':
        test_dataset = Gen1_sbt('/media/HDD3/personal_files/zhanghu/gen1', object_classes='all', height=240, width=304, mode='test', 
                ms_per_frame = args.time_per_frame, frame_per_sequence=args.frame_per_stack, T = args.time_steps, transform=Resize_frame(args.img_size), sbt_method='before')
    else:
        test_dataset = Gen1_sbn('/media/SSD5/personal/zhanghu/gen1/', object_classes='all', height=240, width=304, mode='test', 
                    events_per_frame = args.events_per_frame, frame_per_sequence=args.frame_per_stack, T = args.time_steps, transform=Resize_frame(args.img_size), sbn_method='before')
    classes_name = test_dataset.object_classes
    num_classes = len(classes_name)
    np.random.seed(0)

    # YOLO Config
    cfg = yolov2_tiny_dvs_cfg
    # build model
    from models.yolo.yolov2_tiny_bnn import YOLOv2Tiny_BNN as yolo_net
    model = yolo_net(device=device, 
                   input_size=args.img_size, 
                   num_classes=num_classes, 
                   trainable=False, 
                   cfg=cfg, 
                   center_sample=args.center_sample,
                   time_steps=3,
                   spike_b=args.spike_b,
                   bn=args.bn,
                   init_channels=3,
                   args=args)
    anchor_size = model.anchor_list
    all_keys = [convert_str2index(name,is_cell=True) for name, value in model.named_parameters() if '_ops' in name] 
    all_keys = list(set(all_keys))
    mem_keys = list()
    for key in all_keys:
        try:
            eval('model.%s.mem'%key)
            mem_keys.append(key)
        except:
            print(key)
            pass
    print('mem',mem_keys)
    # model = build_model(args=args, 
    #                     cfg=cfg, 
    #                     device=device, 
    #                     num_classes=num_classes, 
    #                     trainable=False)

    # load weight
    model.load_state_dict(torch.load(args.weight, map_location='cpu'), strict=False)
    model = model.to(device).eval()
    print(np.exp(-1/model.feature.stem0.rho.detach().cpu().numpy().flatten()))
    np.save('alif_tau_a_888.npy', np.exp(-1/model.feature.stem0.rho.detach().cpu().numpy().flatten()))
    input()
    model.training = False
    # for name, layer in model.named_modules():
    #     if isinstance(layer, SNN_2d) or isinstance(layer, SNN_2d_lsnn) or isinstance(layer, SNN_2d_thresh):
    #         layer.fuse_conv_bn()
    model.set_mem_keys(mem_keys)
    print('Finished loading model!')
    gt_label_list = []
    stem0_sparsity = []
    sparsity = {}
    events_sparsity =  []
    with torch.no_grad():
        for index in index_list:
            # print('reset all mem')
            # for key in mem_keys:
            #     exec('model.%s.mem=None'%key)
            image, targets, original_label, original_frame, file = test_dataset[index]
            for label in original_label:
                gt_label_list.append(label)
            # print(targets[0].shape, len(targets[0]))
            targets = [label.tolist() for label in targets]
            size = np.array([[image.shape[-1], image.shape[-2],
                        image.shape[-1], image.shape[-2]]])
            # targets = create_labels.gt_creator(
            #                         img_size=args.img_size, 
            #                         strides=model.stride, 
            #                         label_lists=targets, 
            #                         anchor_size=anchor_size, 
            #                         multi_anchor=args.multi_anchor,
            #                         center_sample=args.center_sample)
            # to device
            image = torch.from_numpy(image).float().to(device).unsqueeze(0)
            # targets = targets.float().to(device)

            # forward
            start_time = time.time()
            conf_pred, cls_pred, reg_pred, box_pred = model.stream_forward_new(image, [file])
            # conf_pred, cls_pred, reg_pred, box_pred = model(image)
            # print('1 sample time:', time.time()-start_time)
            for s in model.input_sparsity:
                events_sparsity.append(s)
            model.input_sparsity = []
            for name, layer in model.named_modules():
                if isinstance(layer, Cell) or isinstance(layer, SNN_2d) or isinstance(layer, SNN_2d_lsnn) or isinstance(layer, BNN_2d) or isinstance(layer, RELU_2d) or isinstance(layer, SNN_2d_thresh):
                    if 'stem0' in name:
                        for s in layer.sparsity:
                            stem0_sparsity.append(s)
                    layer.clear_sparsity()
                        # stem0_sparsity.append(layer.sparsity.numpy()/3)
                    # print(name, type(layer))
                    # if layer.sparsity != None:
                    #     if name not in sparsity.keys():
                    #         sparsity[name] = layer.sparsity.numpy() / 3
                    #     else:
                    #         sparsity[name] = np.append(sparsity[name], layer.sparsity.numpy() / 3)
                    #     layer.sparsity = []
    print(len(stem0_sparsity))
    # plt.plot([60 * (i+1) for i in range(len(stem0_sparsity))], stem0_sparsity)
    # np.save('stem0_5x5_density_lif_2.npy', stem0_sparsity)
    # np.save('stem0_5x5_sparsity_bn_stream.npy', stem0_sparsity)
    np.save('stem0_5x5_sparsity_lif_888.npy', stem0_sparsity)
    # np.save('events_density.npy', events_sparsity)
    # plt.plot([60 * (i+1) for i in range(len(events_sparsity))], events_sparsity)
    # plt.show()
    # plt.savefig('stem0_density_relu.pdf')

        
def plot_result(index_list):
    args = parser.parse_args()
    # device
    if args.device != 'cpu':
        print('use cuda:{}'.format(args.device))
        # cudnn.benchmark = True
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = torch.device("cuda:{}".format(args.device))
    else:
        device = torch.device("cpu")

    model_name = args.model
    print('Model: ', model_name)

    # dataset and evaluator
    if args.frame_method == 'sbt':
        test_dataset = Gen1_sbt('/media/HDD3/personal_files/zhanghu/gen1', object_classes='all', height=240, width=304, mode='test', 
                ms_per_frame = args.time_per_frame, frame_per_sequence=args.frame_per_stack, T = args.time_steps, transform=Resize_frame(args.img_size), sbt_method='before')
    else:
        test_dataset = Gen1_sbn('/media/SSD5/personal/zhanghu/gen1/', object_classes='all', height=240, width=304, mode='test', 
                    events_per_frame = args.events_per_frame, frame_per_sequence=args.frame_per_stack, T = args.time_steps, transform=Resize_frame(args.img_size), sbn_method='before')
    classes_name = test_dataset.object_classes
    num_classes = len(classes_name)
    np.random.seed(0)

    # YOLO Config
    cfg = yolov2_tiny_dvs_cfg
    # build model
    from models.yolo.yolov2_tiny_bnn import YOLOv2Tiny_BNN as yolo_net
    model = yolo_net(device=device, 
                   input_size=args.img_size, 
                   num_classes=num_classes, 
                   trainable=False, 
                   cfg=cfg, 
                   center_sample=args.center_sample,
                   time_steps=3,
                   spike_b=args.spike_b,
                   bn=args.bn,
                   init_channels=3,
                   args=args)
    anchor_size = model.anchor_list
    all_keys = [convert_str2index(name,is_cell=True) for name, value in model.named_parameters() if '_ops' in name] 
    all_keys = list(set(all_keys))
    mem_keys = list()
    for key in all_keys:
        try:
            eval('model.%s.mem'%key)
            mem_keys.append(key)
        except:
            print(key)
            pass
    print('mem',mem_keys)
    # model = build_model(args=args, 
    #                     cfg=cfg, 
    #                     device=device, 
    #                     num_classes=num_classes, 
    #                     trainable=False)

    # load weight
    model.load_state_dict(torch.load(args.weight, map_location='cpu'), strict=False)
    model = model.to(device).eval()
    model.set_mem_keys(mem_keys)
    print('Finished loading model!')
    for name, layer in model.named_modules():
        if isinstance(layer, SNN_2d) or isinstance(layer, SNN_2d_lsnn):
            layer.fuse_conv_bn()
    gt_label_list = []
    idx2class = ['Car', "Ped"]
    idx2color = ['skyblue', 'green']
    plt.ion()
    with torch.no_grad():
        for index in index_list:
            # print('reset all mem')
            for key in mem_keys:
                exec('model.%s.mem=None'%key)
            image, targets, original_label, original_frame, file = test_dataset[index]
            for label in original_label:
                gt_label_list.append(label)
            # print(targets[0].shape, len(targets[0]))
            targets = [label.tolist() for label in targets]
            size = np.array([[image.shape[-1], image.shape[-2],
                        image.shape[-1], image.shape[-2]]])
            # targets = create_labels.gt_creator(
            #                         img_size=args.img_size, 
            #                         strides=model.stride, 
            #                         label_lists=targets, 
            #                         anchor_size=anchor_size, 
            #                         multi_anchor=args.multi_anchor,
            #                         center_sample=args.center_sample)
            # to device
            image = torch.from_numpy(image).float().to(device).unsqueeze(0)
            # targets = targets.float().to(device)

            # forward
            conf_pred, cls_pred, reg_pred, box_pred = model(image)
            bboxes, scores, cls_inds = tools.get_box_score(conf_pred, cls_pred, box_pred, 
                                        num_classes=num_classes, conf_thresh=0.754, nms_thresh=0.5)
            # bboxes *= size
            bboxes = [box * size for box in bboxes]
            bboxes = [tools.resized_box_to_original(box, args.img_size, test_dataset.height, test_dataset.width) for box in bboxes]
            pred_label = []
            for j, (box, score, cls_ind) in enumerate(zip(bboxes[0], scores[0], cls_inds[0])):
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                # label = classes_name[int(cls_ind)]
                
                bbox = [x1, y1, x2 - x1, y2 - y1]
                score = float(score) # object score * class score
                pred_label.append([bbox, cls_ind, score])
            

            print(index)
            original_frame = np.sum(original_frame[-8:, ...], axis=0)
            rgb_frame = 255*np.ones((original_frame.shape[-2], original_frame.shape[-1], 3), dtype=int)
            rgb_frame[original_frame<0, :] = np.array([255, 0, 0])
            rgb_frame[original_frame>0, :] = np.array([0,0,255])
            plt.clf()
            # plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            # plt.subplot(1,2,1)
            plt.imshow(rgb_frame)
            # plt.imshow(original_frame[-1, ...])
            for gt in original_label:
                cls = gt[0]
                x1, y1, w, h = gt[1:]
                # x2, y2 = x1 + w, y1 + h
                lefttop_point = (int(x1), int(y1))
                width_height = (int(w), int(h))
                plt.gca().add_patch(plt.Rectangle(lefttop_point, width_height[0], width_height[1], edgecolor=idx2color[int(cls)], fill=False, linewidth=3))
                plt.gca().text(lefttop_point[0]+2, lefttop_point[1]-3, s='%s'%(idx2class[int(cls)]), color='black', verticalalignment='bottom', fontsize=15, backgroundcolor=idx2color[int(cls)], zorder=5) # backgroundcolor=idx2color[int(cls)]
            plt.savefig('test_0.48_{}_gt.png'.format(index), bbox_inches = 'tight')
            # plt.show()
            # input()
            # plt.subplot(1,2,2)
            plt.clf()
            # plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.imshow(rgb_frame)
            # plt.imshow(original_frame[-1, ...])
            for i in range(len(pred_label)):
                pred = pred_label[i]
                b = pred[0]
                cls = pred[1]
                score = pred[2]
                lefttop_point = (int(b[0]), int(b[1]))
                width_height = (int(b[2]), int(b[3]))
                plt.gca().add_patch(plt.Rectangle(lefttop_point, width_height[0], width_height[1], edgecolor=idx2color[int(cls)], fill=False, linewidth=3))
                if i == 0 and index == 128:
                    plt.gca().text(lefttop_point[0]+2, lefttop_point[1]+ width_height[1]+19 -3, s='%s %.2f'%(idx2class[int(cls)], score), color='black', verticalalignment='bottom', fontsize=15, backgroundcolor=idx2color[int(cls)], zorder=5)
                else:
                    plt.gca().text(lefttop_point[0]+2, lefttop_point[1]-3, s='%s %.2f'%(idx2class[int(cls)], score), color='black', verticalalignment='bottom', fontsize=15, backgroundcolor=idx2color[int(cls)], zorder=5)
            plt.savefig('test_0.48_{}_pred.png'.format(index), bbox_inches = 'tight')
            # plt.show()
            # input()

            # plt.clf()
            # plt.imshow(original_frame[-1, ...], cmap='Greys_r')
            # for pred in pred_label:
            #     b = pred[0]
            #     cls = pred[1]
            #     score = pred[2]
            #     lefttop_point = (int(b[0]), int(b[1]))
            #     width_height = (int(b[2]), int(b[3]))
            #     plt.gca().add_patch(plt.Rectangle(lefttop_point, width_height[0], width_height[1], edgecolor=idx2color[int(cls)], fill=False))
            #     plt.gca().text(lefttop_point[0], lefttop_point[1], s='%s %.2f'%(idx2class[int(cls)], score), color=idx2color[int(cls)], verticalalignment='bottom')
            # plt.savefig('test_0.48_{}.pdf'.format(index))
            # plt.show()
    plt.ioff()

def plot_weight_histogram():
    # fig = plt.figure(figsize=(18, 5))
    lif_stem0_conv_weight_2023 = np.load('lif_stem0_conv_weight_seed2023.npy').flatten()
    alif_stem0_conv_weight_2023 = np.load('alif_stem0_conv_weight_seed2023.npy').flatten()
    lif_stem0_conv_weight = np.load('lif_stem0_conv_weight.npy').flatten()
    alif_stem0_conv_weight = np.load('alif_stem0_conv_weight.npy').flatten()
    bnn_stem0_conv_weight = np.load('bnn_stem0_conv_weight.npy').flatten()
    # plt.hist([lif_stem0_conv_weight, alif_stem0_conv_weight, bnn_stem0_conv_weight], bins=np.linspace(-1.5, 1.5, 30), label=['LIF', 'ALIF', 'BN'])
    fig, ax = plt.subplots(1, 4, figsize=(18, 5), sharex=True, sharey=True)
    ax[0].hist(lif_stem0_conv_weight, bins=100, label='LIF-123')
    ax[0].set_title('LIF-SEED123')
    ax[1].hist(alif_stem0_conv_weight, bins=100, label='ALIF-123')
    ax[1].set_title('ALIF-SEED123')
    ax[2].hist(lif_stem0_conv_weight_2023, bins=100, label='LIF-2023')
    ax[2].set_title('LIF-SEED2023')
    ax[3].hist(alif_stem0_conv_weight_2023, bins=100, label='ALIF-2023')
    ax[3].set_title('ALIF-SEED2023')
    # ax[2].hist(bnn_stem0_conv_weight, bins=100, label='BN')
    # ax[2].set_title('BN')
    # plt.legend(fontsize=18)
    # plt.xticks(fontsize=21)
    # plt.yticks(fontsize=21)
    # plt.y
    # plt.xlabel('pixel', fontsize=21)
    # plt.ylabel('Sparsity', fontsize=21)
    plt.show()

def plot_grad():
    fig = plt.figure(figsize=(6, 5))
    grad_lif_seed888 = np.load('grad_all_lif_seed555.npy')
    grad_alif_seed888 = np.load('grad_all_alif_seed555.npy')
    # fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    print(grad_alif_seed888.shape, grad_lif_seed888.shape)
    plt.plot(list(range(len(grad_lif_seed888))), np.mean(np.abs(grad_lif_seed888), axis=(1, 2, 3, 4)), label='lif', color='b')
    plt.plot(list(range(len(grad_lif_seed888))), np.mean(np.abs(grad_alif_seed888), axis=(1, 2, 3, 4)), label='alif', color='g')
    # ax[0].plot(lif_stem0_conv_weight, bins=100, label='LIF-123')
    # ax[0].set_title('LIF-SEED123')
    # ax[1].hist(alif_stem0_conv_weight, bins=100, label='ALIF-123')
    # ax[1].set_title('ALIF-SEED123')
    # ax[2].hist(lif_stem0_conv_weight_2023, bins=100, label='LIF-2023')
    # ax[2].set_title('LIF-SEED2023')
    # ax[3].hist(alif_stem0_conv_weight_2023, bins=100, label='ALIF-2023')
    # ax[3].set_title('ALIF-SEED2023')
    # ax[2].hist(bnn_stem0_conv_weight, bins=100, label='BN')
    # ax[2].set_title('BN')
    plt.legend(fontsize=18)
    # plt.xticks(fontsize=21)
    # plt.yticks(fontsize=21)
    # plt.y
    # plt.xlabel('pixel', fontsize=21)
    # plt.ylabel('Sparsity', fontsize=21)
    plt.show()

if __name__ == '__main__':
    test()
    # plot_weight_histogram()
    # video_infer()
    # plot_grad()
    # plot_density()
    # plot_dataset_density(list(range(980, 1219)))
    # plot_sparsity()
    # l = [2, 21, 37, 123, 124, 127, 128, 164, 166, 175, 179, 181, 1562, 1564, 1568, 1572, 21756, 21767, 21772, 21774, 30020, 30021, 30022, 30033, 30038, 30041, 30047]
    # l = [128, 164, 21767, 29042]
    # plot_result(l)
    # plot_result([1555])
    # plot_result([30042, 1577, 21764])
    # plot_result(list(range(30020, 30061)))
    # for key, value in sparsity.items():
    #     sparsity[key] = np.mean(value)
    
    # name_list = sparsity.keys()
    # name_list = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 's0', 's1', 'f-c1', 'f-c2', 'f-c3', 'f-u1', 'f-u2', 'f-s1', 'f-s2', 'f-s3']
    # value_list = list(sparsity.values())
    # value_list = value_list[10:12] + value_list[0:10] + value_list[17:]
    # np.save('alif_sparsity.npy', np.array(value_list))
    # name_list = ['s0', 's1', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'f-s1', 'f-s2', 'f-s3']
    # plt.bar(range(len(value_list)), value_list, label='ALIF', tick_label=name_list)
    # plt.show()
    # plt.savefig('alif_sparsity.pdf')
    # print(sparsity)
