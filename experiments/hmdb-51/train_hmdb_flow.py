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
import input_data
import math
import numpy as np
from i3d import InceptionI3d
from utils import *
from tensorflow.python import pywrap_tensorflow

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 20000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 6, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 64, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('rgb_channels', 3, 'RGB_channels for input')
flags.DEFINE_integer('flow_channels', 2, 'FLOW_channels for input')
flags.DEFINE_integer('classics', 101, 'The num of class')
FLAGS = flags.FLAGS
model_save_dir = './models/flow_scratch_20000_6_64_0.0001_decay'

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def run_training():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.

    # Create model directory
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    flow_pre_model_save_dir = "/home/project/I3D/I3D/checkpoints/flow_scratch"

    with tf.Graph().as_default():
        global_step = tf.get_variable(
                        'global_step',
                        [],
                        initializer=tf.constant_initializer(0),
                        trainable=False
                        )
        rgb_images_placeholder, flow_images_placeholder, labels_placeholder, is_training = placeholder_inputs(
                        FLAGS.batch_size * gpu_num,
                        FLAGS.num_frame_per_clib,
                        FLAGS.crop_size,
                        FLAGS.rgb_channels,
                        FLAGS.flow_channels
                        )

        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=5000, decay_rate=0.1, staircase=True)
        opt_flow = tf.train.AdamOptimizer(learning_rate)
        #opt_stable = tf.train.MomentumOptimizer(learning_rate, 0.9)

        with tf.variable_scope('Flow'):
            flow_logit, _ = InceptionI3d(
                                    num_classes=FLAGS.classics,
                                    spatial_squeeze=True,
                                    final_endpoint='Logits'
                                    )(flow_images_placeholder, is_training)
        flow_loss = tower_loss(
                                flow_logit,
                                labels_placeholder
                                )
        accuracy = tower_acc(flow_logit, labels_placeholder)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            flow_grads = opt_flow.compute_gradients(flow_loss)
            apply_gradient_flow = opt_flow.apply_gradients(flow_grads, global_step=global_step)
            train_op = tf.group(apply_gradient_flow)
            null_op = tf.no_op()

        # Create a saver for loading trained checkpoints.
        flow_variable_map = {}

        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow'and 'Adam' not in variable.name.split('/')[-1] and variable.name.split('/')[2] != 'Logits':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph.
        sess = tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True)
                        )
        sess.run(init)
        # Create summary writter
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('flow_loss', flow_loss)
        tf.summary.scalar('learning_rate', learning_rate)
        merged = tf.summary.merge_all()
    # load pre_train models
    ckpt = tf.train.get_checkpoint_state(flow_pre_model_save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        flow_saver.restore(sess, ckpt.model_checkpoint_path)
        print("load complete!")

    train_writer = tf.summary.FileWriter('./visual_logs/train_flow_scratch_20000_6_64_0.0001_decay', sess.graph)
    test_writer = tf.summary.FileWriter('./visual_logs/test_flow_scratch_20000_6_64_0.0001_decay', sess.graph)
    for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        rgb_train_images, flow_train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
                      filename='../../list/hmdb_list/train_flow.list',
                      batch_size=FLAGS.batch_size * gpu_num,
                      num_frames_per_clip=FLAGS.num_frame_per_clib,
                      crop_size=FLAGS.crop_size,
                      shuffle=True
                      )
        sess.run(train_op, feed_dict={
                      flow_images_placeholder: flow_train_images,
                      labels_placeholder: train_labels,
                      is_training: True
                      })
        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (step, duration))

        # Save a checkpoint and evaluate the model periodically.
        if step % 10 == 0 or (step + 1) == FLAGS.max_steps:
            print('Training Data Eval:')
            summary, acc, loss_flow = sess.run(
                            [merged, accuracy, flow_loss],
                            feed_dict={
                                       flow_images_placeholder: flow_train_images,
                                       labels_placeholder: train_labels,
                                       is_training: False
                                      })
            print("accuracy: " + "{:.5f}".format(acc))
            print("flow_loss: " + "{:.5f}".format(loss_flow))
            train_writer.add_summary(summary, step)
            print('Validation Data Eval:')
            rgb_val_images, flow_val_images, val_labels, _, _, _ = input_data.read_clip_and_label(
                            filename='../../list/hmdb_list/test_flow.list',
                            batch_size=FLAGS.batch_size * gpu_num,
                            num_frames_per_clip=FLAGS.num_frame_per_clib,
                            crop_size=FLAGS.crop_size,
                            shuffle=True
                            )
            summary, acc = sess.run(
                            [merged, accuracy],
                            feed_dict={
                                        flow_images_placeholder: flow_val_images,
                                        labels_placeholder: val_labels,
                                        is_training: False
                                      })
            print("accuracy: " + "{:.5f}".format(acc))
            test_writer.add_summary(summary, step)
        if (step+1) % 3000 == 0 or (step + 1) == FLAGS.max_steps:
            saver.save(sess, os.path.join(model_save_dir, 'i3d_hmdb_model'), global_step=step)
    print("done")


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
