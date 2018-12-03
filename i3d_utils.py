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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import math
import input_data
import numpy as np
from multiprocessing import Pool
import threading
from tqdm import tqdm,trange

def placeholder_inputs(batch_size=16, num_frame_per_clib=16, crop_size=224, rgb_channels=3, flow_channels=2):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
    batch_size: The batch size will be baked into both placeholders.
    num_frame_per_clib: The num of frame per clib.
    crop_size: The crop size of per clib.
    channels: The input channel of per clib.

    Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    rgb_images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           num_frame_per_clib,
                                                           crop_size,
                                                           crop_size,
                                                           rgb_channels))
    flow_images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           num_frame_per_clib,
                                                           crop_size,
                                                           crop_size,
                                                           flow_channels))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size
                                                         ))
    is_training = tf.placeholder(tf.bool)
    return rgb_images_placeholder, flow_images_placeholder, labels_placeholder, is_training


def rgb_placeholder_inputs(batch_size=16, num_frame_per_clib=16, crop_size=224, rgb_channels=3, flow_channels=2):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
    batch_size: The batch size will be baked into both placeholders.
    num_frame_per_clib: The num of frame per clib.
    crop_size: The crop size of per clib.
    channels: The input channel of per clib.

    Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    rgb_images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           num_frame_per_clib,
                                                           crop_size,
                                                           crop_size,
                                                           rgb_channels))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size
                                                         ))
    is_training = tf.placeholder(tf.bool)
    return rgb_images_placeholder, labels_placeholder, is_training


def Normalization(clips, frames_num):
    new_clips = []
    for index in range(frames_num):
        clip = tf.image.per_image_standardization(clips[index])
        new_clips.append(clip)
    return new_clips


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def l2_loss(weight_decay, weighyt_list):
    l2_reg = tf.contrib.layers.l2_regularizer(weight_decay)
    return tf.contrib.layers.apply_regularization(regularizer=l2_reg, weights_list=weighyt_list)


def tower_loss(logit, labels, wd):
    print(logit.shape)
    print(labels.shape)
    weight_map = []
    for variable in tf.global_variables():
        if 'conv_3d/w' in variable.name or 'kernel' in variable.name:
            weight_map.append(variable)
    cross_entropy_mean = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logit)
                  )
    weight_decay = l2_loss(wd, weight_map)
    #tf.summary.scalar('sgd_weight_decay_loss', weight_decay)
    # Calculate the total loss for the current tower.
    total_loss = cross_entropy_mean + weight_decay
    return total_loss


def tower_acc(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var)*wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var


def data_to_feed_dict(data):
    rgb_train_images = []
    train_labels = []
    for i in data:
        tmp_train_images = i.get_result()[0]
        tmp_labels = i.get_result()[1]
        rgb_train_images.extend(tmp_train_images)
        train_labels.extend(tmp_labels)
    return np.array(rgb_train_images), np.array(train_labels)


def get_data(filename, batch_size, num_frames_per_clip=64, sample_rate=4, crop_size=224, shuffle=False, add_flow=False):
    rgb_train_images, flow_train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
        filename=filename,
        batch_size=batch_size,
        num_frames_per_clip=num_frames_per_clip,
        sample_rate=sample_rate,
        crop_size=crop_size,
        shuffle=shuffle,
        add_flow=add_flow
    )
    return rgb_train_images, train_labels


class MyThread(threading.Thread):

    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args


    def run(self):

        self.result = self.func(*self.args)


    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def load_data(filename, batch_size, num_frames_per_clip, sample_rate, crop_size, shuffle=False, add_flow=False):
    data = []
    '''
    p = Pool(batch_size/8)
    for i in range(batch_size):
        data.append(p.apply_async(get_data, args=(
            filename,
            8,
            num_frames_per_clip,
            sample_rate,
            crop_size,
            shuffle,
            add_flow
        )))
    p.close()
    #p.join()
    '''
    for i in range(batch_size/4):
        t = MyThread(get_data, args=(
            filename,
            4,
            num_frames_per_clip,
            sample_rate,
            crop_size,
            shuffle,
            add_flow
        ))
        data.append(t)
        t.start()
    for t in data:
        t.join()

    print('DATA_LOAD_COMP: enqueue......')
    rgb_train_images, train_labels = data_to_feed_dict(data)
    return rgb_train_images, train_labels


def topk(predicts, labels, ids):
    scores = {}
    top1_list = []
    top5_list = []
    clips_top1_list = []
    clips_top5_list = []
    start_time = time.time()
    print('Results process..............')
    for index in tqdm(range(len(predicts))):
        id = ids[index]
        score = predicts[index]
        if str(id) not in scores.keys():
            scores['%d'%id] = []
            scores['%d'%id].append(score)
        else:
            scores['%d'%id].append(score)
        avg_pre_index = np.argsort(score).tolist()
        top1 = (labels[id] in avg_pre_index[-1:])
        top5 = (labels[id] in avg_pre_index[-5:])
        clips_top1_list.append(top1)
        clips_top5_list.append(top5)
    print('Clips-----TOP_1_ACC in test: %f' % np.mean(clips_top1_list))
    print('Clips-----TOP_5_ACC in test: %f' % np.mean(clips_top5_list))
    print('..............')
    for _id in range(len(labels)-1):
        avg_pre_index = np.argsort(np.mean(scores['%d'%_id], axis=0)).tolist()
        top1 = (labels[_id] in avg_pre_index[-1:])
        top5 = (labels[_id] in avg_pre_index[-5:])
        top1_list.append(top1)
        top5_list.append(top5)
    print('TOP_1_ACC in test: %f' % np.mean(top1_list))
    print('TOP_5_ACC in test: %f' % np.mean(top5_list))
    duration = time.time() - start_time
    print('Time use: %.3f' % duration)

