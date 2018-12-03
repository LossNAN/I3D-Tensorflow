import os
import math
import numpy as np


root_path = '/home/project/I3D/data/Kinetics/train_256'
num_frames = 16
data_list = []
id_list = []
label_list = []
erro_data = []
label = 0
id = 0

for file_path in sorted(os.listdir(root_path)):
    for video_path in sorted(os.listdir(os.path.join(root_path, file_path))):
        frame_num = len(os.listdir(os.path.join(root_path, file_path, video_path)))
        print('Process: ' + os.path.join(root_path, file_path, video_path), frame_num)
        if frame_num > 0:
            data_list.append(os.path.join(root_path, file_path, video_path))
            id_list.append(id)
            label_list.append(label)
            id += 1
        else:
            erro_data.append(os.path.join(root_path, file_path, video_path))
    label += 1
    if label == 100:
        break
print(erro_data)
print(len(data_list))
print(len(id_list))
print(len(label_list))

np.save('./train_data_list_%d.npy'%label, data_list)
np.save('./train_label_list_%d.npy'%label, label_list)
