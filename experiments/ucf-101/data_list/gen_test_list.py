import os
import math
import numpy as np
from tqdm import tqdm,trange


root_path = '/home/project/I3D/data/Kinetics/val_256'
num_frames = 16
clip_times = 3
data_list = []
id_list = []
position_list = []
label_list = []
erro_data = []
label = 0
id = 0

for file_path in sorted(os.listdir(root_path)):
    for video_path in sorted(os.listdir(os.path.join(root_path, file_path))):
        frame_num = len(os.listdir(os.path.join(root_path, file_path, video_path)))
        print('Process: ' + os.path.join(root_path, file_path, video_path), frame_num)
        if frame_num > 0:
            for start_index in range(int(math.ceil(frame_num / float(num_frames)))):
                for i in range(clip_times):
                    data_list.append([os.path.join(root_path, file_path, video_path),start_index*num_frames])
                    id_list.append(id)
                    position_list.append(i)
            label_list.append(label)
            id += 1
        else:
            erro_data.append(os.path.join(root_path, file_path, video_path))
    label += 1
print('Length of all clips: %d'%len(data_list))
print('Length of all videos: %d'%len(label_list))
np.save('./test_data_list_%d_%dtimes.npy'%(label, clip_times), data_list)
np.save('./test_id_list_%d_%dtimes.npy'%(label, clip_times), id_list)
np.save('./test_label_list_%d_%dtimes.npy'%(label, clip_times), label_list)
np.save('./test_position_list_%d_%dtimes.npy'%(label, clip_times), position_list)