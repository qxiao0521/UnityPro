import tqdm
import sys
sys.path.append('../PyTorch_YOLO-Family')
from data.gen1 import Resize_frame
from data.gen1 import Gen1_sbt
import cv2
import imageio
import numpy as np
import torch
from evaluator.gen1_evaluate import coco_eval
from utils.misc import ori_target_frame_collate

if __name__ == '__main__':
    train_dataset = Gen1_sbt('/media/HDD4/personal_files/liyanchen/gen1', object_classes='all', height=240, width=304, mode='train', 
                ms_per_frame=20, frame_per_sequence=3, T=3, transform=Resize_frame(256), sbt_method='before')
    val_dataset = Gen1_sbt('/media/HDD4/personal_files/liyanchen/gen1', object_classes='all', height=240, width=304, mode='val', 
                ms_per_frame=20, frame_per_sequence=3, T=3, transform=Resize_frame(256), sbt_method='before')
    test_dataset = Gen1_sbt('/media/HDD4/personal_files/liyanchen/gen1', object_classes='all', height=240, width=304, mode='test', 
                ms_per_frame=20, frame_per_sequence=3, T=3, transform=Resize_frame(256), sbt_method='before')

    color = (
        (255, 0, 0),     # image value -1, blue
        (255, 255, 255), # image value  0, white
        (0, 0, 255),     # image value  1, red
        (0, 0, 0),       # image value  2, black
    )
    device = 'cuda:7'

    """select dataset--begin"""
    part = 'test'
    """select dataset--end"""

    if part == 'test':
        dataset = test_dataset
    elif part == 'train':
        dataset = train_dataset
    elif part == 'val':
        dataset = val_dataset
    else:
        dataset = None
    print(f'Generating data from \"{part}\"...')

    '''sparsity statistics and gif generation'''
    img_list = []
    count_list = []
    area_list = []
    sparsity_list = []
    for id_, data in enumerate(tqdm.tqdm(dataset)):
        # if id_ >= 18: break # used for generating gif file

        image, targets, original_label, original_frame, file = data
        image = np.array(image, dtype=int)
        targets = np.array(targets, dtype=int)
        # print(image.shape, np.min(image), np.max(image))
        # print('[class, x, y, w, h]\n', targets)
        # print('[class, x, y, w, h]\n', original_label)

        for target in targets:
            x = target[1]; y = target[2]; w = target[3]; h = target[4]
            x_ = min(x+w, image.shape[2]-1)
            y_ = min(y+h, image.shape[3]-1)
            '''sparsity statistic'''
            area = (x_-x)*(y_-y)
            area_count = np.sum(np.abs(image[:, :, y: y_, x: x_]))
            area_list.append(area)
            count_list.append(area_count)
            sparsity_list.append(area_count/area)
            '''boxes generation'''
            # image[:, :, y: y_, x    ] = 2
            # image[:, :, y: y_, x_   ] = 2
            # image[:, :, y,     x: x_] = 2
            # image[:, :, y_,    x: x_] = 2

        '''gif generation'''
        # for t, chw in enumerate(image):
        #     for c, frame in enumerate(chw):
        #         file_name = f'images/temp_{id_}_{t}_{c}.jpg'
        #         plot_data = np.array([[color[value + 1] for value in frame[row, :]] for row in range(frame.shape[0])])
        #         cv2.imwrite(file_name, plot_data)
        #         img_list.append(imageio.imread(file_name))

        if id_ >= len(dataset) - 1: break

    np.save(f'data/{part}_area.npy', np.array(area_list))
    np.save(f'data/{part}_area_count.npy', np.array(count_list))
    np.save(f'data/{part}_sparsity.npy', np.array(sparsity_list))
    print(f'line 87, {len(img_list)}')
    imageio.mimsave(f'{part}_video.gif', img_list, duration=1e-5)
    print('finished.')
