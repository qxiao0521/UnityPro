# from __future__ import division

# import os
# import random
# import argparse
# import time
# import cv2
# import numpy as np

# import torch
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

# from data.voc0712 import VOCDetection
# from data.coco2017 import COCODataset
# from data.gen1 import Gen1, Resize_frame
# from data import config
# from data import BaseTransform, detection_collate, dvs_detection_collate

# import tools

# from utils import distributed_utils
# from utils.com_paras_flops import FLOPs_and_Params, FLOPs_and_Params_DVS
# from utils.augmentations import SSDAugmentation, ColorAugmentation
# from evaluator.cocoapi_evaluator import COCOAPIEvaluator
# from evaluator.vocapi_evaluator import VOCAPIEvaluator
# from evaluator.gen1_evaluate import coco_eval
# from utils.modules import ModelEMA
# from utils.criterion import build_criterion

# from models import single_gt_model, spiking_model

from __future__ import division

import os
import argparse
import time
import math
import random
import tools
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config.yolo_config import yolo_config
from data.voc import VOCDetection, VOC_CLASSES
from data.coco import COCODataset, coco_class_labels
from data.gen1 import Gen1, Resize_frame, Gen1_sbt, Gen1_sbn
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

from evaluator.cocoapi_evaluator import COCOAPIEvaluator
from evaluator.vocapi_evaluator import VOCAPIEvaluator
from evaluator.gen1_evaluate import coco_eval


# single_gt_model = ['yolov2_d19', 'yolov2_r50', 'yolov2_slim', 'yolov2_tiny', 'yolov2_tiny_fcn', 'yolov2_tiny_relu2spike']
# spiking_model = ['yolov2_tiny_relu2spike']

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--max_epoch', type=int, default=150,
                        help='The upper bound of warm-up')
    parser.add_argument('--lr_epoch', nargs='+', default=[100,200], type=int,
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
    parser.add_argument('--img_size', default=256, type=int,
                        help='img_size')

    # Optimizer & Schedule
    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='sgd, adamw')
    parser.add_argument('--lr_schedule', default='step', type=str,
                        help='step, cos')
    parser.add_argument('--grad_clip', default=None, type=float,
                        help='clip gradient')

    # model
    parser.add_argument('-v', '--version', default='yolo_v2',
                        help='yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny')
    parser.add_argument('-t', '--time_steps', default=5, type=int,
                        help='spike yolo time steps')
    parser.add_argument('-b', '--spike_b', default=3, type=int,
                        help='spike b')
    parser.add_argument('--bn', action='store_false', default=True, 
                        help='use bn layer')
    
    
    # dataset
    parser.add_argument('--root',  default='/media/SSD5/personal/zhanghu/',
                        help='dataset root')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('--conf_thresh', default=0.5, type=float,
                        help='NMS threshold') ## TODO 0.3->0.5
    parser.add_argument('--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    
    # train trick
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use hi-res pre-trained backbone.')  
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')      
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='use mosaic augmentation')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema training trick')
    parser.add_argument('--mixup', action='store_true', default=False,
                        help='use MixUp Augmentation trick')
    parser.add_argument('--multi_anchor', action='store_true', default=False,
                        help='use multiple anchor boxes as the positive samples')
    parser.add_argument('--center_sample', action='store_true', default=False,
                        help='use center sample for labels')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='accumulate gradient')

    # Loss
    parser.add_argument('--loss_obj_weight', default=1.0, type=float,
                        help='weight of obj loss')
    parser.add_argument('--loss_cls_weight', default=1.0, type=float,
                        help='weight of cls loss')
    parser.add_argument('--loss_reg_weight', default=1.0, type=float,
                        help='weight of reg loss')
    parser.add_argument('--scale_loss', default='batch', type=str,
                        help='scale loss: batch or positive samples')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--local_rank', type=int, default=0, 
                        help='local_rank')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')


    ######### LEStereo params ##################
    parser.add_argument('--fea_num_layers', type=int, default=4)
    parser.add_argument('--fea_filter_multiplier', type=int, default=8)
    parser.add_argument('--fea_block_multiplier', type=int, default=3)
    parser.add_argument('--fea_step', type=int, default=3)
    parser.add_argument('--net_arch_fea', default=None, type=str)
    parser.add_argument('--cell_arch_fea', default=None, type=str)
    parser.add_argument('--experiment_description', type=str, default='no description', 
                        help='describ the experiment')    
    parser.add_argument('--fitlog_path',type=str,default='debug')   
    return parser.parse_args()


args = parse_args()

##### FITLOG #####
import git
repo = git.Repo(search_parent_directories=True)
git_branch = path = repo.head.reference.path.split('/')[-1]
git_msg = repo.head.object.summary
git_commit_id = repo.head.object.hexsha[:10]

import fitlog
fitlog_debug = False
if fitlog_debug:
    fitlog.debug()
else:
    fitlog.commit(__file__,fit_msg=args.experiment_description)
    log_path = "logs/rgb_retrain"
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    fitlog.set_log_dir(log_path)
    fitlog.create_log_folder()
    fitlog.add_hyper(args)
    args.fitlog_path = os.path.join(log_path,fitlog.get_log_folder())
    fitlog.add_other(name='git_branch',value=git_branch)
    fitlog.add_other(name='git_msg',value=git_msg)
    fitlog.add_other(name='git_commit_id',value=git_commit_id)

##### end FITLOG #####
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


def train():
    
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

    # set distributed
    local_rank = 0
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = torch.distributed.get_rank()
        print(local_rank)
        torch.cuda.set_device(local_rank)

    # YOLO config
    cfg = yolo_config['yolov3']
    train_size = val_size = args.img_size

    # dataset and evaluator
    train_dataset, val_dataset, evaluator, num_classes = build_dataset(args, train_size, val_size, device)
    # dataloader
    train_dataloader = torch.utils.data.DataLoader(
                        dataset=train_dataset, 
                        shuffle=True,
                        batch_size=args.batch_size, 
                        collate_fn=detection_collate,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    val_dataloader = torch.utils.data.DataLoader(
                        dataset=val_dataset, 
                        shuffle=False,
                        batch_size=args.batch_size, 
                        collate_fn=detection_collate,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    # criterioin
    criterion = build_criterion(args, cfg, num_classes)
    print('Training model on:', args.dataset)
    print('The dataset size:', len(train_dataset))
    print("----------------------------------------------------------")
    model_name = args.version
    print('Model: ', model_name)


    from models.yolo.yolov2_tiny_bnn import YOLOv2Tiny_BNN_RGB as yolo_net
    # path to save model
    # path_to_save = os.path.join(args.save_folder, args.dataset, '{}-{}'.format(args.version, args.time_steps))
    # os.makedirs(path_to_save, exist_ok=True)
    path_to_save = args.fitlog_path + '/'

    # use hi-res backbone
    if args.high_resolution:
        print('use hi-res backbone')
        hr = True
    else:
        hr = False
    
    # # multi-scale
    # if args.multi_scale:
    #     print('use the multi-scale trick ...')
    #     train_size = cfg['train_size']
    #     val_size = cfg['val_size']
    # else:
    #     train_size = val_size = cfg['train_size']
    train_size = val_size = args.img_size
    # Model ENA
    if args.ema:
        print('use EMA trick ...')

    # build model
    anchor_size = cfg['anchor_size_{}'.format(args.dataset)]
    net = yolo_net(device=device, 
                   input_size=train_size, 
                   num_classes=num_classes, 
                   trainable=True, 
                   anchor_size=anchor_size, 
                   center_sample=args.center_sample,
                   time_steps=args.time_steps,
                   spike_b=args.spike_b,
                   bn=args.bn,
                   init_channels=3,
                   args=args)
    model = net

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


    # SyncBatchNorm
    if args.sybn and args.cuda and args.num_gpu > 1:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device)
    params = sum([param.nelement() for param in model.parameters()])
    print('Params : ', params / 1e6, ' M')
    # compute FLOPs and Params
    # FLOPs_and_Params_DVS(model=model, T=args.time_steps, init_channels=args.frame_per_stack, size=train_size, device=device)

    # # distributed
    # if args.distributed and args.num_gpu > 1:
    #     print('using DDP ...')
    #     model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    #     # train_dataloader
    #     train_dataloader = torch.utils.data.DataLoader(
    #                     dataset=train_dataset, 
    #                     batch_size=args.batch_size, 
    #                     collate_fn=dvs_detection_collate,
    #                     num_workers=args.num_workers,
    #                     pin_memory=True,
    #                     sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
    #                     )

    # else:
    #     model = model.train()
    #     # train_dataloader
    #     train_dataloader = torch.utils.data.DataLoader(
    #                     dataset=train_dataset, 
    #                     shuffle=True,
    #                     batch_size=args.batch_size, 
    #                     collate_fn=dvs_detection_collate,
    #                     num_workers=args.num_workers,
    #                     pin_memory=True
    #                     )
    #     # train_dataloader = torch.utils.data.DataLoader(
    #     #                     dataset=train_dataset, 
    #     #                     batch_size=args.batch_size, 
    #     #                     collate_fn=dvs_detection_collate,
    #     #                     num_workers=args.num_workers,
    #     #                     pin_memory=True,
    #     #                     sampler=torch.utils.data.SubsetRandomSampler(train_indices[:32000])
    #     #                     )
    model.train()
    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # EMA
    ema = ModelEMA(model) if args.ema else None

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, '{}-{}-{}-{}-{}-'.format(model_name, args.exp, args.seed, args.lr, args.time_steps) + c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)
    
    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    if args.optimizer == 'sgd':
        print('use SGD with momentum ...')
        optimizer = optim.SGD(model.parameters(), 
                                lr=tmp_lr, 
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        print('use AdamW ...')
        optimizer = optim.AdamW(model.parameters(), 
                                lr=tmp_lr, 
                                weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), 
    #                         lr=base_lr, 
    #                         momentum=args.momentum,
    #                         weight_decay=args.weight_decay
    #                         )

    batch_size = args.batch_size
    # max_epoch = cfg['max_epoch']
    max_epoch = args.max_epoch
    epoch_size = len(train_dataset) // (batch_size * args.num_gpu)

    # criterion
    criterion = build_criterion(args, cfg, num_classes=num_classes)

    best_map = -100.
    warmup = not args.no_warmup

    t0 = time.time()
    # start training loop
    for epoch in range(args.start_epoch, max_epoch):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)        

        # use step lr
        if args.lr_schedule == 'step':
            if epoch in args.lr_epoch:
                tmp_lr = tmp_lr * 0.5
                set_lr(optimizer, tmp_lr)
        # use cos lr decay
        elif args.lr_schedule == 'cos' and not warmup:
            T_max = args.max_epoch - 15
            lr_min = base_lr * 0.1 * 0.1
            if epoch > T_max:
                # Cos decay is done
                print('Cosine annealing is over !!')
                args.lr_schedule == None
                tmp_lr = lr_min
                set_lr(optimizer, tmp_lr)
            else:
                tmp_lr = lr_min + 0.5*(base_lr - lr_min)*(1 + math.cos(math.pi*epoch / T_max))
                set_lr(optimizer, tmp_lr)
        # if epoch in cfg['lr_epoch']:
        #     tmp_lr = tmp_lr * 0.1
        #     set_lr(optimizer, tmp_lr)
    
        for iter_i, (images, targets, _, _) in enumerate(train_dataloader):
            # print('reset all mem')
            for key in mem_keys:
                exec('model.%s.mem=None'%key)


            # WarmUp strategy for learning rate
            ni = iter_i + epoch * epoch_size
            # warmup
            if epoch < args.wp_epoch and warmup:
                nw = args.wp_epoch * epoch_size
                tmp_lr = base_lr * pow(ni / nw, 4)
                set_lr(optimizer, tmp_lr)

            elif epoch == args.wp_epoch and iter_i == 0 and warmup:
                # warmup is over
                print('Warmup is over !!')
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                r = cfg['random_size_range']
                train_size = random.randint(r[0], r[1]) * 32
                model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            targets = [label.tolist() for label in targets]
            # visualize labels
            if args.vis_data:
                vis_data(images, targets)
                continue
            # make labels
            # targets = tools.gt_creator_dvs(input_size=train_size, 
            #                             stride=net.stride, 
            #                             label_lists=targets, 
            #                             anchor_size=anchor_size
            #                             )
            targets = create_labels.gt_creator_rgb(
                                    img_size=train_size, 
                                    strides=net.stride, 
                                    label_lists=targets, 
                                    anchor_size=anchor_size, 
                                    multi_anchor=args.multi_anchor,
                                    center_sample=args.center_sample)
            # visualize assignment
            if args.vis_targets:
                vis_targets(images, targets, anchor_size, net.stride)
                continue
            
            # to device
            images = images.float().to(device)
            targets = targets.float().to(device)

            # forward
            conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred = model(images)
            # conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred, bias_loss = model(images)

            # compute loss
            # conf_loss, cls_loss, box_loss, iou_loss = tools.calculate_loss(conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred, targets)
            # total_loss = conf_loss + cls_loss + box_loss + iou_loss

            conf_loss, cls_loss, box_loss, total_loss = tools.calculate_loss_new(conf_pred, cls_pred, x1y1x2y2_pred, targets, criterion)

            # final_loss = total_loss + bias_loss
            loss_dict = dict(conf_loss=conf_loss,
                             cls_loss=cls_loss,
                             box_loss=box_loss,
                             total_loss=total_loss,
                            #  bias_loss = bias_loss
                            )

            loss_dict_reduced = distributed_utils.reduce_loss_dict(loss_dict)

            # check NAN for loss
            if torch.isnan(total_loss):
                print('nan')
                continue

            # backprop
            total_loss.backward() 
            optimizer.step()
            optimizer.zero_grad() 
            # model.clamp_decay()
            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    tblogger.add_scalar('conf loss',  loss_dict_reduced['conf_loss'].item(),  iter_i + epoch * epoch_size)
                    tblogger.add_scalar('cls loss',  loss_dict_reduced['cls_loss'].item(),  iter_i + epoch * epoch_size)
                    tblogger.add_scalar('box loss',  loss_dict_reduced['box_loss'].item(),  iter_i + epoch * epoch_size)
                    tblogger.add_scalar('total loss',  loss_dict_reduced['total_loss'].item(),  iter_i + epoch * epoch_size)
                    # tblogger.add_scalar('bias loss',  loss_dict_reduced['bias_loss'].item(),  iter_i + epoch * epoch_size)
                    # decay_list = model.get_decay()
                    # tblogger.add_scalars('decay', {'decay{}'.format(i+1): decay_list[i] for i in range(len(decay_list))}, iter_i + epoch * epoch_size)
                    # for name, param in model.named_parameters():  # 返回网络的
                    #     name = name.replace('.', '/')
                    #     tblogger.add_histogram(name, param.data.cpu().numpy(), iter_i + epoch * epoch_size)
                    #     tblogger.add_histogram(name + '/grad', param.grad.cpu().numpy(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                outstream = ('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                        '[Loss: conf %.2f || cls %.2f || box %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, 
                           max_epoch, 
                           iter_i, 
                           epoch_size, 
                           tmp_lr,
                           loss_dict_reduced['conf_loss'].item(),
                           loss_dict_reduced['cls_loss'].item(), 
                           loss_dict_reduced['box_loss'].item(),
                           loss_dict_reduced['total_loss'].item(),
                           train_size, 
                           t1-t0))
                # outstream = ('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                #         '[Loss: conf %.2f || cls %.2f || box %.2f || bias %.2f || total %.2f || size %d || time: %.2f]'
                #         % (epoch+1, 
                #            max_epoch, 
                #            iter_i, 
                #            epoch_size, 
                #            tmp_lr,
                #            loss_dict_reduced['conf_loss'].item(),
                #            loss_dict_reduced['cls_loss'].item(), 
                #            loss_dict_reduced['box_loss'].item(),
                #            loss_dict_reduced['bias_loss'].item(),
                #            loss_dict_reduced['total_loss'].item(),
                #            train_size, 
                #            t1-t0))

                print(outstream, flush=True)

                t0 = time.time()
            # ema
            if args.ema:
                ema.update(model)

            # display
            if fitlog_debug:
                print('!!!!')
                break
        if (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == args.max_epoch:
            if args.ema:
                model_eval = ema.ema
            else:
                model_eval = model.module if args.distributed else model

            # set eval mode
            model_eval.trainable = False
            model_eval.set_grid(val_size)
            model_eval.eval()
            batch_num = len(val_dataloader)
            conf_loss_list = []
            cls_loss_list = []
            box_loss_list = []
            bias_loss_list = []
            total_loss_list = []
            gt_label_list = []
            pred_label_list = []
            if args.dataset == 'voc':
                classes_name = VOC_CLASSES
            elif args.dataset == 'coco':
                classes_name = coco_class_labels
            start_time = time.time()
            conf_out_of_range = 0
            box_out_of_range = 0
            with torch.no_grad():
                for id_, data in enumerate(val_dataloader):
                    # print('reset all mem')
                    for key in mem_keys:
                        exec('model.%s.mem=None'%key)

                    image, targets, height, width = data
                    for ids in range(len(targets)):
                        temp = []
                        for label in targets[ids]:
                            temp.append(tools.resized_label_to_original(label, val_size, height[ids], width[ids]))
                        gt_label_list.append(temp)
                    # print(targets[0].shape, len(targets[0]))
                    targets = [label.tolist() for label in targets]
                    size = np.array([[image.shape[-1], image.shape[-2],
                                image.shape[-1], image.shape[-2]]])
                    targets = create_labels.gt_creator(
                                            img_size=val_size, 
                                            strides=net.stride, 
                                            label_lists=targets, 
                                            anchor_size=anchor_size, 
                                            multi_anchor=args.multi_anchor,
                                            center_sample=args.center_sample)
                    # to device
                    image = image.float().to(device)
                    targets = targets.float().to(device)

                    # forward
                    conf_pred, cls_pred, reg_pred, box_pred = model_eval(image)
                    # conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred, bias_loss = model_eval(image)
                    
                    # calculate loss
                    # conf_loss, cls_loss, box_loss, iou_loss = tools.calculate_loss(conf_pred, cls_pred, txtytwth_pred, x1y1x2y2_pred, targets)
                    # total_loss = conf_loss + cls_loss + box_loss + iou_loss

                    conf_loss, cls_loss, box_loss, total_loss = tools.calculate_loss_new(conf_pred, cls_pred, box_pred, targets, criterion)

                    # final_loss = total_loss + bias_loss
                    conf_loss_list.append(conf_loss)
                    cls_loss_list.append(cls_loss)
                    box_loss_list.append(box_loss)
                    # bias_loss_list.append(bias_loss)
                    total_loss_list.append(total_loss)

                    bboxes, scores, cls_inds = tools.get_box_score(conf_pred, cls_pred, box_pred, 
                                                num_classes=num_classes, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh)
                    # print(len(box))
                    # bboxes *= size
                    bboxes = [box * size for box in bboxes]
                    bboxes = [tools.resized_box_to_original(bboxes[i], val_size, height[i], width[i]) for i in range(len(bboxes))]
                    
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
            conf_loss_item = sum(conf_loss_list).item() / batch_num
            cls_loss_item = sum(cls_loss_list).item() / batch_num
            box_loss_item = sum(box_loss_list).item() / batch_num
            total_loss_item = sum(total_loss_list).item() / batch_num
            print('val/conf loss', conf_loss_item)
            print('val/cls loss',  cls_loss_item)
            print('val/box loss',  box_loss_item)
            print('val/total loss',  total_loss_item)
            if fitlog_debug:
                map50_95, map50 = 100, 100
            else:
                map50_95, map50 = coco_eval(gt_label_list, pred_label_list, height=240, width=304, labelmap=classes_name)
            cur_map = map50
            if args.dataset == 'voc':
                print('val mAP(0.5): {}'.format(map50))
            elif args.dataset == 'coco':
                print('val mAP(0.5):{} mAP(0.5:0.95): {}'.format(map50, map50_95))
            
            if fitlog_debug:
                break 
              
            print('val mAP(0.5:0.95):{}, mAP(0.5):{}'.format(map50_95, map50))
            # print('val/conf pred (< 0 or > 1): {},  box pred (< 0 or > 1): {}'.format(
            #     conf_out_of_range, box_out_of_range))
            if args.tfboard:
                tblogger.add_scalar('val/conf loss',  conf_loss_item,  epoch)
                tblogger.add_scalar('val/cls loss',  cls_loss_item,  epoch)
                tblogger.add_scalar('val/box loss',  box_loss_item,  epoch)
                # tblogger.add_scalar('val/bias loss',  bias_loss_item,  epoch)
                tblogger.add_scalar('val/total loss',  total_loss_item,  epoch)
                tblogger.add_scalar('val mAP(0.5:0.95)', map50_95, epoch)
                tblogger.add_scalar('val mAP(0.5)', map50, epoch)
                # tblogger.add_scalar('val/conf pred (< 0 or > 1)',  conf_out_of_range,  epoch)
                # tblogger.add_scalar('val/box pred (< 0 or > 1)',  box_out_of_range,  epoch)
            if args.dataset == 'voc':
                fitlog.add_best_metric(name="Valid Best mAP(0.5)",value=map50)
            elif args.dataset == 'coco':
                fitlog.add_best_metric(name="Valid Best mAP(0.5)",value=map50)
                fitlog.add_best_metric(name="Valid Best mAP(0.5:0.95)",value=map50_95)

            if cur_map > best_map:
                # update best-map
                best_map = cur_map
                if args.dataset == 'voc':
                    fitlog.add_best_metric(name="Valid Best mAP(0.5)",value=map50)
                elif args.dataset == 'coco':
                    fitlog.add_best_metric(name="Valid Best mAP(0.5)",value=map50)
                    fitlog.add_best_metric(name="Valid Best mAP(0.5:0.95)",value=map50_95)
                # save model
                print('Saving state, epoch:', epoch + 1)
                torch.save(model_eval.state_dict(), os.path.join(path_to_save, 
                            args.version + '_' + repr(epoch + 1) + '_' + str(round(best_map, 2)) + '.pth')
                            )

            # wait for all processes to synchronize
            if args.distributed:
                dist.barrier()

            # set train mode.
            model_eval.trainable = True
            model_eval.set_grid(train_size)
            model_eval.train()

            if fitlog_debug:
                break 
    if args.tfboard:
        tblogger.close()

def build_dataset(args, train_size, val_size, device):
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        num_classes = 20
        train_dataset = VOCDetection(
                        data_dir=data_dir,
                        img_size=train_size,
                        transform=TrainTransforms(train_size),
                        color_augment=ColorTransforms(train_size),
                        mosaic=args.mosaic,
                        mixup=args.mixup)
        val_dataset = VOCDetection(data_dir=data_dir, 
                                    img_size = val_size,
                                    image_sets=[('2007', 'test')],
                                    transform=ValTransforms(val_size))
        evaluator = VOCAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=ValTransforms(val_size))

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'coco2017')
        num_classes = 80
        train_dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=train_size,
                    image_set='train2017',
                    transform=TrainTransforms(train_size),
                    color_augment=ColorTransforms(train_size),
                    mosaic=args.mosaic,
                    mixup=args.mixup)
        val_dataset = COCODataset(data_dir=data_dir,
                                    img_size=val_size,
                                    image_set='val2017',
                                    transform=ValTransforms(val_size))
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=ValTransforms(val_size)
                        )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    return train_dataset, val_dataset, evaluator, num_classes


def build_dataloader(args, dataset, collate_fn=None):
    # distributed
    if args.distributed and args.num_gpu > 1:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        sampler=torch.utils.data.distributed.DistributedSampler(dataset)
                        )

    else:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        shuffle=True,
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    return dataloader

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def vis_data(images, targets, input_size):
    # vis data
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = images[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    img = ((img * std + mean)*255).astype(np.uint8)
    img = img.copy()

    for box in targets[0]:
        xmin, ymin, xmax, ymax = box[:-1]
        # print(xmin, ymin, xmax, ymax)
        xmin *= input_size
        ymin *= input_size
        xmax *= input_size
        ymax *= input_size
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    train()
