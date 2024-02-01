from __future__ import division

import os
import argparse
import time
import math
import random
import tools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config.yolo_config import yolov2_tiny_dvs_cfg
from data.voc import VOCDetection
from data.coco import COCODataset
from data.gen1 import Gen1, Resize_frame, Gen1_sbt
from data.transforms import TrainTransforms, ColorTransforms, ValTransforms

from utils import distributed_utils
from utils import create_labels
from utils.vis import vis_data, vis_targets
from utils.com_flops_params import FLOPs_and_Params, FLOPs_and_Params_DVS
from utils.criterion import build_criterion
from utils.misc import detection_collate, dvs_detection_collate
from utils.misc import ModelEMA
from utils.criterion import build_criterion

from models.yolo import build_model
from models.yolo.yolov2_tiny_bnn import Conv_Bn_LeakyReLu, Conv_Bn_LTC, Conv_Bn_Spike

from evaluator.cocoapi_evaluator import COCOAPIEvaluator
from evaluator.vocapi_evaluator import VOCAPIEvaluator
from evaluator.gen1_evaluate import coco_eval


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--max_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--lr_epoch', nargs='+', default=[90, 120], type=int,
                        help='lr epoch to decay')
    parser.add_argument('--wp_epoch', type=int, default=1,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--num_gpu', default=1, type=int, 
                        help='Number of GPUs to train')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')
    parser.add_argument('--vis_data', action='store_true', default=False,
                        help='visualize images and labels.')
    parser.add_argument('--vis_targets', action='store_true', default=False,
                        help='visualize assignment.')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed')
    parser.add_argument('--exp', default='normal', type=str,
                        help='description of the experiment')
    parser.add_argument('--weight', default='weight/',
                    type=str, help='Trained state_dict file path to open')

    # Optimizer & Schedule
    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='sgd, adamw')
    parser.add_argument('--lr_schedule', default='step', type=str,
                        help='step, cos')
    parser.add_argument('--grad_clip', default=None, type=float,
                        help='clip gradient')
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')

    # model
    parser.add_argument('-v', '--version', default='yolo_v2',
                        help='yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny')
    parser.add_argument('-t', '--time_steps', default=2, type=int,
                        help='spike yolo time steps')
    parser.add_argument('-tf', '--time_per_frame', default=125, type=int,
                        help='spike yolo time steps')
    parser.add_argument('-fs', '--frame_per_stack', default=1, type=int,
                        help='spike yolo time steps')
    parser.add_argument('-b', '--spike_b', default=3, type=int,
                        help='spike b')
    parser.add_argument('--bn', action='store_false', default=True, 
                        help='use bn layer')
    
    
    # dataset
    parser.add_argument('-root', '--data_root', default='/home/z50021442/object_detection/datasets',
                        help='dataset root')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('--conf_thresh', default=0.3, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')

    # Loss

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--local_rank', type=int, default=0, 
                        help='local_rank')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()

def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = False  # 保证每次返回得的卷积算法是确定
    if args.device != 'cpu':
        print('use cuda:{}'.format(args.device))
        # cudnn.benchmark = True
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        device = torch.device("cuda:{}".format(args.device))
    else:
        device = torch.device("cpu")
    
    model_name = args.version
    print('Model: ', model_name)

    # load model and config file
    if model_name == 'yolov2_tiny_bnn':
        from models.yolo.yolov2_tiny_bnn import YOLOv2Tiny_BNN as yolo_net
    elif model_name == 'yolov2_tiny_ann':
        from models.yolo.yolov2_tiny_bnn import YOLOv2Tiny_ANN as yolo_net
    elif model_name == 'yolov2_tiny_snn_bnn':
        from models.yolo.yolov2_tiny_bnn import YOLOv2Tiny_SNN_BNN as yolo_net
    elif model_name == 'yolov2_tiny_snn_ann':
        from models.yolo.yolov2_tiny_bnn import YOLOv2Tiny_SNN_ANN as yolo_net
    elif model_name == 'yolov2_tiny_ann_fit':
        from models.yolo.yolov2_tiny_bnn import YOLOv2Tiny_ANN_Fit as yolo_net
    elif model_name == 'yolov2_tiny_snnm_ann':
        from models.yolo.yolov2_tiny_bnn import YOLOv2Tiny_SNNM_ANN as yolo_net
    elif model_name == 'yolov2_tiny_snn_ann_fit':
        from models.yolo.yolov2_tiny_bnn import YOLOv2Tiny_SNN_ANN_Fit as yolo_net
    elif model_name == 'yolov2_tiny_ltc_ann_fit':
        from models.yolo.yolov2_tiny_bnn import YOLOv2Tiny_LTC_ANN_Fit as yolo_net
    else:
        from models.yolo.yolov2_tiny_bnn import YOLOv2Tiny_BNN as yolo_net
    cfg = yolov2_tiny_dvs_cfg

    # path to save model
    # path_to_save = os.path.join(args.save_folder, args.dataset, '{}-{}'.format(args.version, args.time_steps))
    # os.makedirs(path_to_save, exist_ok=True)

    train_size = val_size = cfg['train_size']

    num_classes = 2
    train_dataset = Gen1_sbt('/media/SSD5/personal/zhanghu/gen1', object_classes='all', height=240, width=304, mode='train', 
                ms_per_frame = args.time_per_frame, frame_per_sequence=args.frame_per_stack, T = args.time_steps, transform=Resize_frame(train_size), sbt_method='mid')
    test_dataset = Gen1_sbt('/media/SSD5/personal/zhanghu/gen1', object_classes='all', height=240, width=304, mode='test', 
                ms_per_frame = args.time_per_frame, frame_per_sequence=args.frame_per_stack, T = args.time_steps, transform=Resize_frame(val_size), sbt_method='mid')
    
    anchor_size = cfg['anchor_size_gen1']
    model = yolo_net(device=device, 
                   input_size=train_size, 
                   num_classes=num_classes, 
                   trainable=True, 
                   anchor_size=anchor_size, 
                   center_sample=False,
                   time_steps=args.time_steps,
                   spike_b=args.spike_b,
                   bn=args.bn,
                   init_channels=args.frame_per_stack)
    model.load_state_dict(torch.load(args.weight, map_location='cpu'), strict=False)
    model.to(device)

    

    # fig, ax = plt.subplots(1, 6)
    # record_conv_bias(model, ax, 'before_fuse')
    model.eval()
    
    train_dataloader = torch.utils.data.DataLoader(
                        dataset=train_dataset,
                        shuffle=True,
                        batch_size=args.batch_size, 
                        collate_fn=dvs_detection_collate,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        )
    test_dataloader = torch.utils.data.DataLoader(
                        dataset=test_dataset, 
                        batch_size=args.batch_size, 
                        collate_fn=dvs_detection_collate,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        )
    for name, m in model.named_modules():
        print(name)
    for name, m in model.named_parameters():
        print(name)
    evaluate(model, test_dataloader, val_size, train_dataset.object_classes, args.conf_thresh, args.nms_thresh, anchor_size, device, num_classes)
    model = fuse_model(model, device)
    model.to(device)
    # for name, m in model.named_modules():
    #     print(name)
    # input()
    model = model_bias_2_weight(model, device)
    model.to(device)
    for name, m in model.named_modules():
        print(name)
    for name, m in model.named_parameters():
        print(name)
    evaluate_special(model, test_dataloader, val_size, train_dataset.object_classes, args.conf_thresh, args.nms_thresh, anchor_size, device, num_classes)
    # ex_bias_parameters = model.get_exclude_bias_parameters()
    # bias_parameters = model.get_bias_parameters()
    # # print([param.shape for param in ex_bias_parameters])
    # # print([param.shape for param in bias_parameters])
    # # input()
    # base_lr = args.lr
    # tmp_lr = base_lr
    # if args.optimizer == 'sgd':
    #     print('use SGD with momentum ...')
    #     optimizer = optim.SGD(model.parameters(), 
    #                             lr=args.lr, 
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    # elif args.optimizer == 'adamw':
    #     print('use AdamW ...')
    #     optimizer = optim.AdamW(model.parameters(), 
    #                             lr=args.lr,
    #                             weight_decay=args.weight_decay)

    
    # batch_size = args.batch_size
    # epoch_size = len(train_dataloader) // (args.num_gpu)

    # warmup = not args.no_warmup
    # model.train()
    # t0 = time.time()
    # for epoch in range(args.start_epoch, args.max_epoch):
    #     if args.distributed:
    #         train_dataloader.sampler.set_epoch(epoch)        

    #     # use step lr
    #     if args.lr_schedule == 'step':
    #         if epoch in args.lr_epoch:
    #             tmp_lr = tmp_lr * 0.1
    #             set_lr(optimizer, tmp_lr)
    #     # use cos lr decay
    #     elif args.lr_schedule == 'cos' and not warmup:
    #         T_max = args.max_epoch - 15
    #         lr_min = base_lr * 0.1 * 0.1
    #         if epoch > T_max:
    #             # Cos decay is done
    #             print('Cosine annealing is over !!')
    #             args.lr_schedule == None
    #             tmp_lr = lr_min
    #             set_lr(optimizer, tmp_lr)
    #         else:
    #             tmp_lr = lr_min + 0.5*(base_lr - lr_min)*(1 + math.cos(math.pi*epoch / T_max))
    #             set_lr(optimizer, tmp_lr)
    
    #     for iter_i, (images, targets) in enumerate(train_dataloader):
    #         # WarmUp strategy for learning rate
    #         ni = iter_i + epoch * epoch_size
    #         # warmup
    #         # if epoch < args.wp_epoch and warmup:
    #         #     nw = args.wp_epoch * epoch_size
    #         #     tmp_lr = base_lr * pow(ni / nw, 4)
    #         #     set_lr(optimizer, tmp_lr)

    #         # elif epoch == args.wp_epoch and iter_i == 0 and warmup:
    #         #     # warmup is over
    #         #     print('Warmup is over !!')
    #         #     warmup = False
    #         #     tmp_lr = base_lr
    #         #     set_lr(optimizer, tmp_lr)
            
    #         targets = [label.tolist() for label in targets]
    #         targets = create_labels.gt_creator(
    #                                 img_size=train_size, 
    #                                 strides=model.stride, 
    #                                 label_lists=targets, 
    #                                 anchor_size=anchor_size, 
    #                                 multi_anchor=False,
    #                                 center_sample=False)
    #         # visualize assignment
    #         if args.vis_targets:
    #             vis_targets(images, targets, anchor_size, model.stride)
    #             continue
            
    #         # to device
    #         images = images.float().to(device)
    #         targets = targets.float().to(device)

    #         conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred, bias_loss = model(images)


    #         # backprop
    #         bias_loss.backward()
    #         # final_loss.backward()        
    #         optimizer.step()
    #         # model.clamp_decay()
    #         if iter_i % 10 == 0:
    #             t1 = time.time()
    #             outstream = ('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
    #                     '[bias loss: %.2f || size %d || time: %.2f]'
    #                     % (epoch+1, 
    #                        args.max_epoch, 
    #                        iter_i, 
    #                        epoch_size, 
    #                        tmp_lr,
    #                        bias_loss.item(),
    #                        train_size, 
    #                        t1-t0))
    #             print(outstream, flush=True)

    #             t0 = time.time()
    #         optimizer.zero_grad()
    # model.eval()
    # model = fuse_model(model, device)
    # model.to(device)
    # record_conv_bias(model, ax, 'after_fuse')
    # for i in range(len(ax)):
    #     ax[i].set_xlabel('channel')
    #     ax[i].set_ylabel('value')
    #     ax[i].set_title('conv{}'.format(i+1))
    #     ax[i].legend()
    # evaluate(model, test_dataloader, val_size, train_dataset.object_classes, args.conf_thresh, args.nms_thresh, anchor_size, device, num_classes)
    # plt.show()

def fuse_conv_and_bn(conv, bn):
    # 初始化
    with torch.no_grad():
        fusedconv = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True
        )
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        if bn.weight != None:
            w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
        else:
            w_bn = torch.diag(1 / torch.sqrt(bn.eps+bn.running_var))
        # 融合层的权重初始化(W_bn*w_conv(卷积的权重))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        if conv.bias != None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))
        
        if bn.bias != None:    
            b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        else:
            b_bn = -bn.running_mean.div(torch.sqrt(bn.running_var + bn.eps))
        # 融合层偏差的设置
        fusedconv.bias.copy_(torch.matmul(w_bn, b_conv) + b_bn)

    return fusedconv


def fuse_model(model, device):
    for m in model.modules():
        if isinstance(m, Conv_Bn_LeakyReLu):
            conv = m.layer[0]
            if not isinstance(m.layer[1], torch.nn.BatchNorm2d):
                continue
            bn = m.layer[1]
            # bn.weight = torch.nn.para torch.ones(1)
            # bn.weight = torch.nn.Parameter(torch.nn.init.constant_(torch.Tensor(1), 1).to(device))
            # # bn.bias = torch.zeros(1)
            # bn.bias = torch.nn.Parameter(torch.nn.init.constant_(torch.Tensor(1), 0).to(device))
            # print(bn, bn.weight, bn.bias, bn.running_mean)
            fuse_conv = fuse_conv_and_bn(conv, bn)
            m.layer[0] = fuse_conv
            del m.layer[1]
    return model

def convert_bias_2_weight(conv, device, add_one_out_channel=True):
    inchannels = conv.in_channels
    outchannels = conv.out_channels
    kernel_size=conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    
    if conv.bias == None:
        bias = torch.zeros((outchannels))
    else:
        bias = conv.bias.data
    weight = conv.weight.data
    b_weight = torch.zeros((outchannels, 1, kernel_size[0], kernel_size[1]))
    for i in range(outchannels):
        b_weight[i, :, kernel_size[0]//2, kernel_size[1]//2] = bias[i]
    b_weight = b_weight.to(device)
    if add_one_out_channel == True:
        add_weight = torch.zeros((1, inchannels+1, kernel_size[0], kernel_size[1]))
        add_weight[:, 0, kernel_size[0]//2, kernel_size[1]//2] = 1
        add_weight = add_weight.to(device)
        conv.weight = torch.nn.Parameter(torch.cat((add_weight, torch.cat((b_weight, weight), 1)), 0))
    else:
        conv.weight = torch.nn.Parameter(torch.cat((b_weight, weight), 1))
    # add_weight = add_weight.to(device)
    # conv.weight = nn.Parameter(torch.cat((weight, b_weight), 1))
    # conv.weight = torch.nn.Parameter(torch.cat((add_weight, torch.cat((b_weight, weight), 1)), 0))
    conv.bias = None
    return conv

def convert_pred(pred_conv, device):
    inchannels = pred_conv.in_channels
    outchannels = pred_conv.out_channels
    kernel_size=pred_conv.kernel_size
    stride=pred_conv.stride
    padding=pred_conv.padding

    new_outchannels = pow(2, math.ceil(math.log2(outchannels)))
    add_weight = torch.zeros((new_outchannels-outchannels, inchannels+1, kernel_size[0], kernel_size[1]))
    
    if pred_conv.bias == None:
        bias = torch.zeros((outchannels))
    else:
        bias = pred_conv.bias.data
    weight = pred_conv.weight.data
    b_weight = torch.zeros((outchannels, 1, kernel_size[0], kernel_size[1]))
    for i in range(outchannels):
        b_weight[i, :, kernel_size[0]//2, kernel_size[1]//2] = bias[i]
    b_weight = b_weight.to(device)
    if new_outchannels - outchannels != 0:
        add_weight = torch.zeros((new_outchannels-outchannels, inchannels+1, kernel_size[0], kernel_size[1]))
        # add_weight[:, 0, kernel_size[0]//2, kernel_size[1]//2] = 1
        add_weight = add_weight.to(device)
        pred_conv.weight = torch.nn.Parameter(torch.cat((torch.cat((b_weight, weight), 1), add_weight), 0))
    else:
        pred_conv.weight = torch.nn.Parameter(torch.cat((b_weight, weight), 1))
    pred_conv.bias = None
    new_conv = torch.nn.Conv2d(new_outchannels, outchannels*16, 1, 1, bias=False)
    new_conv_weight = torch.zeros((outchannels*16, new_outchannels, 1, 1))
    for i in range(outchannels):
        new_conv_weight[i * 16, i, :, :] = 1
    new_conv.weight = torch.nn.Parameter(new_conv_weight)

    return torch.nn.Sequential(pred_conv, new_conv)

    # add_weight = add_weight.to(device)
    # conv.weight = nn.Parameter(torch.cat((weight, b_weight), 1))
    # conv.weight = torch.nn.Parameter(torch.cat((add_weight, torch.cat((b_weight, weight), 1)), 0))
    
    # return conv


def model_bias_2_weight(model, device):
    for name, m in model.named_modules():
        if isinstance(m, Conv_Bn_LeakyReLu):
            conv = m.layer[0]
            # if conv.bias == None:
            #     continue
            convert_conv = convert_bias_2_weight(conv, device)
            m.layer[0] = convert_conv
        elif name == 'pred':
            # m = convert_bias_2_weight(m, device, False)
            model.pred = convert_pred(m, device)
            # m = convert_pred(m, device)
    return model

def record_conv_bias(model, ax, exp):
    i = 0
    for name, m in model.named_modules():
        if isinstance(m, Conv_Bn_LeakyReLu):
            conv = m.layer[0]
            if not m.bn:
                continue
            ax[i].plot(range(1, len(conv.bias)+1), conv.bias.cpu().detach().numpy(), label=exp)
            i += 1
    return ax






def evaluate(model, dataloader, val_size, classes_name, conf_thresh, nms_thresh, anchor_size, device, num_classes):
    model.trainable = False
    model.set_grid(val_size)
    batch_num = len(dataloader)
    gt_label_list = []
    pred_label_list = []
    start_time = time.time()
    with torch.no_grad():
        for id_, data in enumerate(dataloader):
            image, targets = data
            for target in targets:
                gt_label_list.append(target)
            # print(targets[0].shape, len(targets[0]))
            targets = [label.tolist() for label in targets]
            size = np.array([[image.shape[-1], image.shape[-2],
                        image.shape[-1], image.shape[-2]]])
            targets = create_labels.gt_creator(
                                    img_size=val_size, 
                                    strides=model.stride, 
                                    label_lists=targets, 
                                    anchor_size=anchor_size, 
                                    multi_anchor=False,
                                    center_sample=False)
            # to device
            image = image.float().to(device)
            targets = targets.float().to(device)

            # forward
            # conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred = model(image)
            conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred, bias_loss = model(image)

            bboxes, scores, cls_inds = tools.get_box_score(conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred, 
                                        num_classes=num_classes, conf_thresh=conf_thresh, nms_thresh=nms_thresh)

            bboxes = [box * size for box in bboxes]
            
            for i in range(len(bboxes)):
                pred_label = []
                for j, (box, score, cls_ind) in enumerate(zip(bboxes[i], scores[i], cls_inds[i])):
                    x1 = float(box[0])
                    y1 = float(box[1])
                    x2 = float(box[2])
                    y2 = float(box[3])
                    # label = classes_name[int(cls_ind)]
                    
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    score = float(score) # object score * class score
                    A = {"image_id": id_ * 64 + i, "category_id": cls_ind, "bbox": bbox,
                        "score": score} # COCO json format
                    pred_label.append(A)
                pred_label_list.append(pred_label)
    # print(gt_label_list[0:5])
    # print(pred_label_list[0:5])
    # print('inference time(batch size = 1):{}'.format(time.time()-start_time))
    map50_95, map50 = coco_eval(gt_label_list, pred_label_list, height=val_size, width=val_size, labelmap=classes_name)
    cur_map = map50_95

    print('test/bias loss',  bias_loss.item())
    print('test mAP(0.5:0.95):{}, mAP(0.5):{}'.format(map50_95, map50))

def evaluate_special(model, dataloader, val_size, classes_name, conf_thresh, nms_thresh, anchor_size, device, num_classes):
    model.trainable = False
    model.set_grid(val_size)
    batch_num = len(dataloader)
    gt_label_list = []
    pred_label_list = []
    start_time = time.time()
    with torch.no_grad():
        for id_, data in enumerate(dataloader):
            image, targets = data
            B, T, C, H, W = image.shape
            for target in targets:
                gt_label_list.append(target)
            # print(targets[0].shape, len(targets[0]))
            targets = [label.tolist() for label in targets]
            size = np.array([[image.shape[-1], image.shape[-2],
                        image.shape[-1], image.shape[-2]]])
            targets = create_labels.gt_creator(
                                    img_size=val_size, 
                                    strides=model.stride, 
                                    label_lists=targets, 
                                    anchor_size=anchor_size, 
                                    multi_anchor=False,
                                    center_sample=False)
            # to device
            image = torch.cat((torch.ones((B, 1, C, H, W)), image), 1).float().to(device)
            targets = targets.float().to(device)

            # forward
            # conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred = model(image)
            B, T, C, H, W = image.shape
            x = image.view(B, -1, H, W)
            x = model.down_sample1(model.conv1(x))
            x = model.down_sample2(model.conv2(x))
            x = model.down_sample3(model.conv3(x))
            x = model.conv4(x)

            x = model.conv5(x)
            x = model.conv6(x)
            x = model.pred(x)
            B, abC, H, W = x.size()
            final_pred = x[:, ::16, :, :].permute(0, 2, 3, 1).contiguous().view(B, H * W, -1)
            # [B, H*W*num_anchor, 1]
            conf_pred = final_pred[:, :, :1 * model.num_anchors].contiguous().view(B, H * W * model.num_anchors, 1)
            # [B, H*W*num_anchor, num_cls]
            cls_pred = final_pred[:, :, 1 * model.num_anchors: (1 + model.num_classes) * model.num_anchors].contiguous().view(
                B, H * W * model.num_anchors, model.num_classes)
            # [B, H*W, num_anchor, 4]
            txtytwth_pred = final_pred[:, :, (1 + model.num_classes) * model.num_anchors:(1+model.num_classes+4) * model.num_anchors].contiguous().view(B, H * W, model.num_anchors, 4)
            # [B, H*W*num_anchor, 4]
            # 该步将原本的坐标预测从相对于grid的(0,1)数据准换为相对于整个图片的(0,1)数据
            x1y1x2y2_pred = model.decode_bbox(txtytwth_pred) / model.input_size


            # conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred, bias_loss = model(image)

            bboxes, scores, cls_inds = tools.get_box_score(conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred, 
                                        num_classes=num_classes, conf_thresh=conf_thresh, nms_thresh=nms_thresh)

            bboxes = [box * size for box in bboxes]
            
            for i in range(len(bboxes)):
                pred_label = []
                for j, (box, score, cls_ind) in enumerate(zip(bboxes[i], scores[i], cls_inds[i])):
                    x1 = float(box[0])
                    y1 = float(box[1])
                    x2 = float(box[2])
                    y2 = float(box[3])
                    # label = classes_name[int(cls_ind)]
                    
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    score = float(score) # object score * class score
                    A = {"image_id": id_ * 64 + i, "category_id": cls_ind, "bbox": bbox,
                        "score": score} # COCO json format
                    pred_label.append(A)
                pred_label_list.append(pred_label)
    # print(gt_label_list[0:5])
    # print(pred_label_list[0:5])
    # print('inference time(batch size = 1):{}'.format(time.time()-start_time))
    map50_95, map50 = coco_eval(gt_label_list, pred_label_list, height=val_size, width=val_size, labelmap=classes_name)
    cur_map = map50_95

    # print('test/bias loss',  bias_loss.item())
    print('test mAP(0.5:0.95):{}, mAP(0.5):{}'.format(map50_95, map50))

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    train()