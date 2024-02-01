SAVE_PATH='/home/z50021442/snn_darts_obj_detect/PyTorch_YOLO-Family/weights/voc/yolov2_tiny_bnn_search-2/'
C:\\Soft\\Anaconda-2020.11\\envs\\pytorch_env\\python test_dvs.py \
--experiment_description 'test' \
-m yolov2_tiny_bnn \
--frame_method sbt \
-tf 20 -fs 1 -t 10 \
--batch_size 64 \
--img_size 256 \
--device 6 \
--fea_num_layers 10 \
--net_arch_fea=''$SAVE_PATH'feature_network_path.npy' \
--cell_arch_fea=''$SAVE_PATH'feature_genotype.npy' \
--fea_filter_multiplier 32 \
--multi_anchor \
--conf_thresh 0.3 \
--weight "/media/HDD2/personal_files/zhanghu/PyTorch_YOLO-Family_logs/logs/retrain/log_20230308_151419/yolov2_tiny_bnn_28_0.47.pth"

# "/media/HDD2/personal_files/zhanghu/PyTorch_YOLO-Family_logs/logs/retrain/log_20230307_131023/yolov2_tiny_bnn_22_0.47.pth" stem0 5x5 alif seed=2023
# "/media/HDD2/personal_files/zhanghu/PyTorch_YOLO-Family_logs/logs/retrain/log_20230307_131051/yolov2_tiny_bnn_29_0.47.pth" stem0 5x5 lif tau=0.2 seed=2023
# "/media/HDD2/personal_files/zhanghu/PyTorch_YOLO-Family_logs/logs/retrain/log_20230308_093818/yolov2_tiny_bnn_29_0.47.pth" all lif tau=0.2 seed999
# "/media/HDD2/personal_files/zhanghu/PyTorch_YOLO-Family_logs/logs/retrain/log_20230308_093909/yolov2_tiny_bnn_29_0.47.pth" alif tau=0.2 seed999
#  alif tau=0.2 seed888
# "/media/HDD2/personal_files/zhanghu/PyTorch_YOLO-Family_logs/logs/retrain/log_20230308_151528/yolov2_tiny_bnn_26_0.46.pth" all lif tau=0.2 seed888


# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221111_102223/yolov2_tiny_bnn_23_0.444.pth" 5000ef S=2 C=5 stem0 5x5 all lif
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221111_103530/yolov2_tiny_bnn_24_0.445.pth" 5000ef S=5 C=3 _> S=3 C=3 stem0 5x5 all lif
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221111_105413/yolov2_tiny_bnn_29_0.47.pth" 20ms S=10 C=1 _> S=2 C=5 stem0 5x5 all lif
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221111_105519/yolov2_tiny_bnn_25_0.474.pth" 20ms S=10 C=1 _> S=2 C=5 stem0 5x5 alif
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221111_160557/yolov2_tiny_bnn_22_0.393.pth" resnet_snn [2,3,3,2] init_C=76


#  -ef 5000 -fs 3 -t 5 \


# "/home/z50021442/snn_darts_obj_detect/PyTorch_YOLO-Family/logs/retrain/log_20220627_101517/yolov2_tiny_bnn_20_0.42.pth"
# "/home/z50021442/snn_darts_obj_detect/PyTorch_YOLO-Family/logs/retrain/log_20220706_141448/yolov2_tiny_bnn_24_0.36.pth"
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20220907_175329/yolov2_tiny_bnn_30_0.46.pth"
# "/media/HDD2/personal_files/zhanghu/PyTorch_YOLO-Family_logs/logs/retrain/log_20221104_085841/yolov2_tiny_bnn_29_0.48.pth" stem0 5x5 alif
# "/media/HDD2/personal_files/zhanghu/PyTorch_YOLO-Family_logs/logs/retrain/log_20221107_111807/yolov2_tiny_bnn_24_0.46.pth" stem0 5x5 bn
# "/media/HDD2/personal_files/zhanghu/PyTorch_YOLO-Family_logs/logs/retrain/log_20221104_085808/yolov2_tiny_bnn_25_0.47.pth" stem0 5x5 lif tau=0.2

# "/media/HDD2/personal_files/zhanghu/PyTorch_YOLO-Family_logs/logs/retrain/log_20230305_141331/yolov2_tiny_bnn_28_0.45.pth" stem0 5x5 lif tau=0.1
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221105_133645/yolov2_tiny_bnn_29_0.46.pth" resnet_snn init_C=96
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221105_133517/yolov2_tiny_bnn_17_0.45.pth" resnet_snn init_C=64
# "/media/HDD2/personal_files/zhanghu/PyTorch_YOLO-Family_logs/logs/retrain/log_20221105_131739/yolov2_tiny_bnn_30_0.43.pth" all bnn
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221109_220135/yolov2_tiny_bnn_30_0.436.pth" resnet_snn init_C=80
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221108_221821/yolov2_tiny_bnn_24_0.424.pth" resnet_snn init_C=64
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221108_221848/yolov2_tiny_bnn_30_0.451.pth" resnet_snn init_C=96
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221110_111050/yolov2_tiny_bnn_24_0.472.pth" stem0 5x5 alif cudnn=False x x x
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221110_130623/yolov2_tiny_bnn_27_0.464.pth" stem0 lif thre=0.4169
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221106_200029/yolov2_tiny_bnn_28_0.42.pth" all alif

# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221110_110432/yolov2_tiny_bnn_24_0.47.pth" stem0 5x5 alif rho fixed 0.87
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221110_111050/yolov2_tiny_bnn_24_0.472.pth" stem0 5x5 alif rho learnable 0.87
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221104_090334/yolov2_tiny_bnn_26_0.48.pth" stem0 3x3 relu x x x
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221104_090152/yolov2_tiny_bnn_25_0.47.pth" stem0 3x3 lif  x 0.82 0.845
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221104_090224/yolov2_tiny_bnn_30_0.47.pth" stem0 3x3 alif x x x
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221104_090057/yolov2_tiny_bnn_20_0.45.pth" init_C=40 stem0 3x3 lif x x x
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221104_090034/yolov2_tiny_bnn_28_0.48.pth" init_C=40 stem0 3x3 alif x x x
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221105_125139/yolov2_tiny_bnn_28_0.49.pth" init_C=40 stem0 3x3 lif x x x 
# "/home/z50021442/PyTorch_YOLO-Family/logs/retrain/log_20221105_132017/yolov2_tiny_bnn_10_0.52.pth" stem0 3x3 all relu x x x
