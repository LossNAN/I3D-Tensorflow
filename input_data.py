# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for load train data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time


def get_data(filename, num_frames_per_clip=64, s_index=-1):
    ret_arr = []
    filenames = ''
    for parent, dirnames, filenames in os.walk(filename):
        if len(filenames) < num_frames_per_clip:
            filenames = sorted(filenames)
            for i in range(num_frames_per_clip):
                if i >= len(filenames):
                    i = i % len(filenames)
                image_name = str(filename) + '/' + str(filenames[i])
                img = Image.open(image_name)
                img_data = np.array(img)
                ret_arr.append(img_data)
            return ret_arr, s_index
    filenames = sorted(filenames)
    if s_index < 0:
        s_index = random.randint(0, len(filenames) - num_frames_per_clip)
    for i in range(s_index, s_index + num_frames_per_clip):
        image_name = str(filename) + '/' + str(filenames[i])
        img = Image.open(image_name)
        img_data = np.array(img)
        ret_arr.append(img_data)
    return ret_arr, s_index


def get_frames_data(filename, num_frames_per_clip=64):
    ''' Given a directory containing extracted frames, return a video clip of
    (num_frames_per_clip) consecutive frames as a list of np arrays '''
    filename_i = os.path.join(filename, 'i')
    rgb_ret_arr, s_index = get_data(filename_i, num_frames_per_clip)
    filename_x = os.path.join(filename, 'x')
    flow_x, _ = get_data(filename_x, num_frames_per_clip, s_index)
    flow_x = np.expand_dims(flow_x, axis=-1)
    filename_y = os.path.join(filename, 'y')
    flow_y, _ = get_data(filename_y, num_frames_per_clip, s_index)
    flow_y = np.expand_dims(flow_y, axis=-1)
    flow_ret_arr = np.concatenate((flow_x, flow_y), axis=-1)
    return rgb_ret_arr, flow_ret_arr, s_index


def data_process(tmp_data, crop_size):
    img_datas = []
    for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
            scale = float(crop_size) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
            scale = float(crop_size) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        img_datas.append(img)
    return img_datas


def read_clip_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=64, crop_size=224, shuffle=False):
    lines = open(filename, 'r')
    read_dirnames = []
    rgb_data = []
    flow_data = []
    label = []
    batch_index = 0
    next_batch_start = -1
    lines = list(lines)
    # Forcing shuffle, if start_pos is not specified
    if start_pos < 0:
        shuffle = True
    if shuffle:
        video_indices = range(len(lines))
        random.seed(time.time())
        random.shuffle(video_indices)
    else:
        # Process videos sequentially
        video_indices = range(start_pos, len(lines))
    for index in video_indices:
        if batch_index >= batch_size:
            next_batch_start = index
            break
        line = lines[index].strip('\n').split()
        dirname = line[0]
        tmp_label = line[1]
        if not shuffle:
            print("Loading a video clip from {}...".format(dirname))
        tmp_rgb_data, tmp_flow_data, _ = get_frames_data(dirname, num_frames_per_clip)
        if len(tmp_rgb_data) != 0:
            rgb_img_datas = data_process(tmp_rgb_data, crop_size)
            flow_img_datas = data_process(tmp_flow_data, crop_size)
            rgb_data.append(rgb_img_datas)
            flow_data.append(flow_img_datas)
            label.append(int(tmp_label))
            batch_index = batch_index + 1
            read_dirnames.append(dirname)

    # pad (duplicate) data/label if less than batch_size
    valid_len = len(rgb_data)
    pad_len = batch_size - valid_len
    if pad_len:
        for i in range(pad_len):
            rgb_data.append(rgb_img_datas)
            flow_data.append(flow_img_datas)
            label.append(int(tmp_label))

    np_arr_rgb_data = np.array(rgb_data).astype(np.float32)
    np_arr_flow_data = np.array(flow_data).astype(np.float32)
    np_arr_label = np.array(label).astype(np.int64)

    return np_arr_rgb_data, np_arr_flow_data, np_arr_label.reshape(batch_size), next_batch_start, read_dirnames, valid_len
