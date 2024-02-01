import pandas as pd
import numpy as np
import imageio
import os
import sys
import argparse

def event2frame_SBE_txyp(event_stream_us, stack_duration_us, width, height, event_rate=0.35):
    # preprocessing
    event_stream_us[:, 0] -= event_stream_us[0, 0]
    event_stream_us[:, 3][event_stream_us[:, 3] == 0] = -1
    Ne = width*height*event_rate


def event2frame_SBT_txyp(event_stream_us, stack_duration_us, width, height, frame_per_stack=1):
    # preprocessing
    event_stream_us[:, 0] -= event_stream_us[0, 0]
    event_stream_us[:, 3][event_stream_us[:, 3] == 0] = -1

    step = stack_duration_us/frame_per_stack # [0, step), [step, 2*step), [2*step, 3*step) ...
    left = 0
    right = step
    max_time = event_stream_us[event_stream_us.shape[0] - 1, 0]
    frame_list = np.zeros(shape=(max_time//stack_duration_us + 1, height, width), dtype=int)
    frame_pointer = 0

    for index in range(event_stream_us.shape[0]):
        time_point = event_stream_us[index, 0]
        coord_x = event_stream_us[index, 1]
        coord_y = event_stream_us[index, 2]
        polarity = event_stream_us[index, 3]

        if time_point >= right:
            left += step
            right += step
            frame_pointer += 1

        frame_list[frame_pointer][coord_y][coord_x] += polarity

    for index in range(len(frame_list)):
        frame_list[index][frame_list[index] > 0] = 1
        frame_list[index][frame_list[index] < 0] = 1 # single color, original: -1

    return frame_list

def show_sign_frames_plt(output_dir, frame_list):
    import matplotlib
    import matplotlib.pyplot as plt

    image_list = list()
    plt.ion()
    flag = False
    for index in range(len(frame_list)):
        frame = frame_list[index].copy().astype(np.float64)
        # draw a frame
        up = 50
        down = 550
        left = 500
        right = 900
        frame[up: down, left] = np.nan
        frame[up: down, right] = np.nan
        frame[up, left: right] = np.nan
        frame[down, left: right] = np.nan
        plt.clf()
        c_map = plt.cm.get_cmap('gray_r').copy()
        c_map.set_bad('r')
        matplotlib.rc("font", family='Microsoft YaHei')
        plt.title('识别结果：%d\n' % (index), fontsize=20)
        plt.imshow(frame, cmap=c_map)
        temp_file_name = os.path.join(output_dir, '_%d.jpg' % (index))
        plt.savefig(temp_file_name)
        image_list.append(plt.imread(temp_file_name))
        plt.pause(1e-10)
    plt.ioff()

    return image_list

def show_sign_frames_cv2(output_dir, frame_list, inteval_per_frame_ms=1):
    import cv2
    image_list = list()
    color = [
        [255, 0, 0],     # -1, blue
        [255, 255, 255], #  0, white
        [0, 0, 255],     #  1, red
    ]

    for index in range(len(frame_list)):
        frame = frame_list[index]
        plot_data = np.array([
            [color[value + 1] for value in frame[row, :]] for row in range(frame.shape[0])
        ])
        temp_file_name = os.path.join(output_dir, '_.jpg')
        cv2.imwrite(temp_file_name, plot_data)
        image_list.append(imageio.imread(temp_file_name))
        cv2.imshow('test', cv2.imread(temp_file_name))
        cv2.waitKey(inteval_per_frame_ms)

    return image_list

def count_sparsity(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    count = 0
    for i in range(height):
        for j in range(width):
            if frame[i][j] != 0:
                count += 1
    return count / height / width


if __name__ == '__main__':
    '''
    ## preload the image data
    import cv2
    from pathlib import Path
    image_path = '../#dataset/IJRR/calibration/images'
    name_list = [Path(item).as_posix() for item in os.scandir(image_path) if item.is_file()]
    gt_list = np.zeros((len(name_list), 180, 240), dtype=float)
    for index, name in enumerate(name_list):
        gt_list[index] = cv2.imread(name, cv2.IMREAD_GRAYSCALE) / 255.0
    np.save('./proc_IJRR/groundtruth_data/calibration.npy', gt_list)
    sys.exit(1)
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir', type=str, default='../#dataset/IJRR/calibration')
    parser.add_argument('-output_dir', type=str, default='D:\\PytorchPro\\Data\\Gen1\\train_a\\gen1\\sbt_20ms_1frame_10sequence\\output')
    parser.add_argument('-input_mode', type=str, default='calibration_2000us_per_frame')
    args = parser.parse_args()

    # mode_txt = 'events.txt'
    # mode_npy = ''.join([args.input_mode, '.npy'])
    # mode_gif = ''.join([args.input_mode, '.gif'])
    #
    # ## 1. extract the numpy integer "txyp" vectors (microsecond) from "events.txt"
    # source_dir = os.path.join(args.input_dir, mode_txt)
    # with open(source_dir, 'r') as fp:
    #     lines = fp.readlines()
    #     event_stream = np.zeros((len(lines), 4), float)
    #     for index, line in enumerate(lines):
    #         event_stream[index] = np.array(line.split(), float)
    #     event_stream[:, 0] = event_stream[:, 0]*1e6  # convert second to microsecond
    #     event_stream = event_stream.astype(int)
    # max_x = np.max(event_stream[:, 1])
    # max_y = np.max(event_stream[:, 2])
    # print('The max X:', max_x)
    # print('The max Y:', max_y)
    #
    # ## 2. framing the event_stream into image tensors
    # frame_list = event2frame_SBT_txyp(
    #     event_stream_us = event_stream,
    #     stack_duration_us = 2000, # Unit: microsecond
    #     width = max_x + 1,
    #     height = max_y + 1,
    #     frame_per_stack = 1
    # )
    # # save frame result as ".npy" format
    # npy_name = os.path.join(args.output_dir, mode_npy)
    # np.save(npy_name, frame_list)
    # print('Framing Finished.')
    #sys.exit(1)
    frame_list_all = []
    for i in range(30):
        npy_name = f'D:\\PytorchPro\\Data\\Gen1\\train_a\\gen1\\sbt_20ms_1frame_10sequence\\train\\sample{i}_frame.npy'
    ## 3. test and store the visualization result
        frame_list = np.load(npy_name)
        frame_list_all.extend(frame_list[:, 0])
    # print(frame_list.shape)
    # import sys; sys.exit(1)
    image_list = show_sign_frames_cv2(args.output_dir, frame_list_all)
    # generate the gif
    # gif_name = os.path.join(args.output_dir, mode_gif)
    # imageio.mimsave(gif_name, image_list, duration=0.00005)
    print('Visualization finished.')

    '''
    # draw the sparsity chart
    x = range(len(frame_list))
    y = [count_sparsity(frame_list[i]) for i in range(len(frame_list))]
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.plot(x, y, mec='r', mfc='w',label=u'y=x^2')
    plt.plot(x, y, marker='*', ms=10,label=u'y=x^3')
    plt.legend()
    plt.title('test')
    plt.show()
    '''
