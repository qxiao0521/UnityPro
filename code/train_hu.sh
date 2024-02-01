SAVE_PATH='/home/z50029642/results/weights/voc/yolov2_tiny_bnn_search-2/'
~/anaconda3/bin/python train_dvs.py \
--experiment_description 'lsnn 20ms s=10, c=1 _> s=3, c=3 all relu tau learnable' \
-v yolov2_tiny_bnn \
-tf 20 -fs 1 -t 10 \
--batch_size 32 \
--lr 1e-3 \
--device 7 \
--max_epoch 30 \
--eval_epoch 1 \
--optimizer adamw \
--fea_num_layers 10 \
--net_arch_fea=''$SAVE_PATH'feature_network_path.npy' \
--cell_arch_fea=''$SAVE_PATH'feature_genotype.npy' \
--fea_filter_multiplier 32 \
--fea_block_multiplier 3 \
--fea_step 3 \
--multi_anchor \
--conf_thresh 0.3


# --resume '/home/z50021442/snn_darts_obj_detect/PyTorch_YOLO-Family/logs/retrain/log_20220822_093152/yolov2_tiny_bnn_30_0.43.pth'
