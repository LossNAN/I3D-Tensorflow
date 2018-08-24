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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import math
import numpy as np
from i3d import InceptionI3d
from utils import *

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 2
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 16, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('channels', 3, 'Channels for input')
flags.DEFINE_integer('classics', 101, 'The num of class')
FLAGS = flags.FLAGS


def run_test():
    model_name = ""
    test_list_file = 'list/ucf_list/test.list'
    num_test_videos = len(list(open(test_list_file, 'r')))
    print("Number of test videos={}".format(num_test_videos))
    # Get the sets of images and labels for training, validation, and
    images_placeholder, labels_placeholder = placeholder_inputs(
                                            FLAGS.batch_size * gpu_num,
                                            FLAGS.batch_size * gpu_num,
                                            FLAGS.num_frame_per_clib,
                                            FLAGS.crop_size,
                                            FLAGS.channels
    )
    logits = []
    for gpu_index in range(0, gpu_num):
        with tf.device('/gpu:%d' % gpu_index):
            i3d_tuple = InceptionI3d(
                num_classes=FLAGS.classics,
                spatial_squeeze=True,
                final_endpoint='Logits',
                name='inception_i3d'
            )(images_placeholder[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size, :, :, :, :],
              is_training=False)
            logit = i3d_tuple[0]
            logits.append(logit)
    logits = tf.concat(logits, 0)
    norm_score = tf.nn.softmax(logits)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)
    # Create a saver for writing training checkpoints.
    saver.restore(sess, model_name)
    # And then after everything is built, start the training loop.
    bufsize = 0
    write_file = open("predict_ret.txt", "w+", bufsize)
    next_start_pos = 0
    all_steps = int((num_test_videos - 1) / (FLAGS.batch_size * gpu_num) + 1)
    for step in xrange(all_steps):
        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        start_time = time.time()
        test_images, test_labels, next_start_pos, _, valid_len = input_data.read_clip_and_label(
                        test_list_file,
                        FLAGS.batch_size * gpu_num,
                        start_pos=next_start_pos
                        )
        predict_score = norm_score.eval(
                session=sess,
                feed_dict={images_placeholder: test_images}
                )
        for i in range(0, valid_len):
            true_label = test_labels[i],
            top1_predicted_label = np.argmax(predict_score[i])
            # Write results: true label, class prob for true label, predicted label, class prob for predicted label
            write_file.write('{}, {}, {}, {}\n'.format(
                  true_label[0],
                  predict_score[i][true_label],
                  top1_predicted_label,
                  predict_score[i][top1_predicted_label]))
    write_file.close()
    print("done")


def main(_):
    run_test()


if __name__ == '__main__':
    tf.app.run()
