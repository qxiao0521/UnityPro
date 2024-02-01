# YOLO config


yolo_config = {
    'yolov1': {
        # backbone
        'backbone': 'r50',
        # neck
        'neck': 'dilated_encoder',
        # anchor size
        'anchor_size': None
    },
    'yolov2': {
        # backbone
        'backbone': 'r50',
        # neck
        'neck': 'dilated_encoder',
        # anchor size: P5-640
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolov3': {
        # backbone
        'backbone': 'd53',
        # neck
        'neck': 'conv_blocks',
        # anchor size: P5-640
        'anchor_size_voc': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolov3_spp': {
        # backbone
        'backbone': 'd53',
        # neck
        'neck': 'spp',
        # anchor size: P5-640
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolov3_de': {
        # backbone
        'backbone': 'd53',
        # neck
        'neck': 'dilated_encoder',
        # anchor size: P5-640
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolov4': {
        # backbone
        'backbone': 'cspd53',
        # neck
        'neck': 'spp',
        # anchor size: P5-640
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, qfl
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolov5_s': {
        # backbone
        'backbone': 'csp_s',
        'width': 0.5,
        'depth': 0.33,
        'depthwise': False,
        'freeze': False,
        # neck
        'neck': 'yolopafpn',
        # head
        'head_dim': 256,
        # anchor size: P5-640
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, bce
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolov5_m': {
        # backbone
        'backbone': 'csp_m',
        'width': 0.75,
        'depth': 0.67,
        'depthwise': False,
        'freeze': False,
        # neck
        'neck': 'yolopafpn',
        # head
        'head_dim': 256,
        # anchor size: P5-640
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, bce
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolov5_l': {
        # backbone
        'backbone': 'csp_l',
        'width': 1.0,
        'depth': 1.0,
        'depthwise': False,
        'freeze': False,
        # neck
        'neck': 'yolopafpn',
        # head
        'head_dim': 256,
        # anchor size: P5-640
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, bce
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolov5_x': {
        # backbone
        'backbone': 'csp_x',
        'width': 1.25,
        'depth': 1.33,
        'depthwise': False,
        'freeze': False,
        # neck
        'neck': 'yolopafpn',
        # head
        'head_dim': 256,
        # anchor size: P5-640
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, bce
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolov5_t': {
        # backbone
        'backbone': 'csp_t',
        'width': 0.375,
        'depth': 0.33,
        'depthwise': False,
        'freeze': False,
        # neck
        'neck': 'yolopafpn',
        # head
        'head_dim': 256,
        # anchor size: P5-640
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, bce
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolov5_n': {
        # backbone
        'backbone': 'csp_n',
        'width': 0.25,
        'depth': 0.33,
        'depthwise': True,
        'freeze': False,
        # neck
        'neck': 'yolopafpn',
        # head
        'head_dim': 256,
        # anchor size: P5-640
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, bce
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },    
    'yolo_tiny': {
        # backbone
        'backbone': 'cspd_tiny',
        # neck
        'neck': 'spp-csp',
        # anchor size: P5-640
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolo_nano': {
        # backbone
        'backbone': 'sfnet_v2',
        # neck
        'neck': 'spp-dw',
        # anchor size: P5-640
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, qfl
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolo_nano_plus': {
        # backbone
        'backbone': 'csp_n',
        'depthwise': True,
        # neck
        'neck': 'yolopafpn',
        # anchor size: P5-640
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, qfl
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolotr': {
        # backbone
        'backbone': 'vit_b',
        # neck
        'neck': 'dilated_encoder',
        # anchor size: P5-640
        'anchor_size_coco': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    }
}

# YOLOv2Tiny
yolov2_tiny_dvs_cfg = {
    # network
    'backbone': 'dlight',
    # for multi-scale trick
    'train_size': 320,
    'val_size': 320,
    'random_size_range': [10, 19],
    # anchor size
    # 'anchor_size_gen1': [[0.95, 0.69], [0.49, 1.38], [1.41, 1.02], [2.24, 1.48], [3.19, 2.47]],
    # 'anchor_size_gen1': [[31, 22], [18, 49], [45, 32], [70, 47], [106, 81]],
    # 'anchor_size_gen1': [[28,21], [42,30], [65,43], [101,76]],
    # 'anchor_size_gen1': [[29,22], [50,35], [92,69]],
    # input_size 304x240
    'anchor_size_gen1':[[14,38],   [28,20],   [36,27],
                        [25,64],   [49,34],   [66,44],
                        [85,62],  [107,82], [138,120]],
    # input_size 256
    'anchor_size_gen1_9':[[12,33],   [25,17],   [33,24],
                        [23,58],   [46,31],   [63,42],
                        [79,58],  [97,78], [147,112]],
    'anchor_size_gen1_6':[[14,38],   [26,18],   [40,28],
                        [59,44],   [85,63],   [119,97]],
    'anchor_size_gen1_3':[[26,23],   [52,37],   [90,69]],
    # train
    'lr_epoch': (90, 120),
    'max_epoch': 150,
    'ignore_thresh': 0.5
}