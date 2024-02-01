import numpy as np
import matplotlib.pyplot as plt
import re

parts = ('test', 'val', 'train')
root_dir = './lyc20231012'

timestep = 3
channel = 3
test_mAP = []
with open('./data/mAP_record.txt', 'r') as fp:
    for line_num, line in enumerate(fp):
        # test_mAP.extend([float(re.findall(r'mAP\(0.5\): \d+\.\d+', line)[0].split(' ')[1]) for _ in range(timestep*channel)])
        test_mAP.append(float(re.findall(r'mAP\(0.5\): \d+\.\d+', line)[0].split(' ')[1]))
test_mAP = np.array(test_mAP)

for part in parts:
    area = np.load(f'{root_dir}/data/{part}_area.npy')
    count = np.load(f'{root_dir}/data/{part}_area_count.npy')
    sparsity = np.load(f'{root_dir}/data/{part}_sparsity.npy')
    print(len(sparsity), len(area), len(count), len(test_mAP))

    lb = 0
    ub = min(100, len(area))
    # print(len(area))
    t = np.arange(lb, ub)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(t, area[lb: ub], label='area')
    plt.plot(t, count[lb: ub], label='count')
    plt.plot(t, test_mAP[lb: ub]*np.max(area[lb: ub]), label='mAP(50)')
    plt.xlabel('frame')
    plt.ylabel('value')
    plt.legend()
    plt.title(f'\"{part}\", frame in [{lb}, {ub})')

    plt.subplot(1, 2, 2)
    plt.plot(t, sparsity[lb: ub], label='sparsity')
    plt.plot(t, test_mAP[lb: ub]*np.max(sparsity[lb: ub]), label='mAP(50)')
    plt.xlabel('frame')
    plt.ylabel('value')
    plt.legend()
    plt.title(f'sparsity of \"{part}\", frame in [{lb}, {ub})')
    plt.savefig(f'{root_dir}/{part}.png', dpi=600)

    '''only in test dataset'''
    break
