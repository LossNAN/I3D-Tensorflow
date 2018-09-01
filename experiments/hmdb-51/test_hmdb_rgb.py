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
import input_test
import math
import numpy as np
from i3d import InceptionI3d
from utils import *

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 16, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'Channels for input')
flags.DEFINE_integer('classics', 101, 'The num of class')
FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def run_training():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.
    pre_model_save_dir = "./models/rgb_imagenet_10000_6_64_0.0001_decay"
    test_list_file = '../../list/hmdb_list/test_flow.list'
    file = list(open(test_list_file, 'r'))
    num_test_videos = len(file)
    print("Number of test videos={}".format(num_test_videos))
    with tf.Graph().as_default():
        rgb_images_placeholder, _, labels_placeholder, is_training = placeholder_inputs(
                        FLAGS.batch_size * gpu_num,
                        FLAGS.num_frame_per_clib,
                        FLAGS.crop_size,
                        FLAGS.rgb_channels
                        )

        with tf.variable_scope('RGB'):
            logit, _ = InceptionI3d(
                                num_classes=FLAGS.classics,
                                spatial_squeeze=True,
                                final_endpoint='Logits',
                                name='inception_i3d'
                                )(rgb_images_placeholder, is_training)
        norm_score = tf.nn.softmax(logit)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph.
        sess = tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True)
                        )
        sess.run(init)

    ckpt = tf.train.get_checkpoint_state(pre_model_save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print ("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("load complete!")

    all_steps = num_test_videos
    top1_list = []
    for step in xrange(all_steps):
        start_time = time.time()
        s_index = 0
        predicts = []
        top1 = False
        while True:
            val_images, _, val_labels, s_index, is_end = input_test.read_clip_and_label(
                            filename=file[step],
                            batch_size=FLAGS.batch_size * gpu_num,
                            s_index=s_index,
                            num_frames_per_clip=FLAGS.num_frame_per_clib,
                            crop_size=FLAGS.crop_size,
                            )
            predict = sess.run(norm_score,
                               feed_dict={
                                            rgb_images_placeholder: val_images,
                                            labels_placeholder: val_labels,
                                            is_training: False
                                            })
            predicts.append(np.array(predict).astype(np.float32).reshape(FLAGS.classics))
            if is_end:
                avg_pre = np.mean(predicts, axis=0).tolist()
                top1 = (avg_pre.index(max(avg_pre))==val_labels)
                top1_list.append(top1)
                break
        duration = time.time() - start_time
        print('TOP_1_ACC in test: %f , time use: %.3f' % (top1, duration))
    print(len(top1_list))
    print('TOP_1_ACC in test: %f' % np.mean(top1_list))
    print("done")


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
