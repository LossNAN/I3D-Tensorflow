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

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 16, 'Nummber of frames per clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('channels', 3, 'Channels for input')
flags.DEFINE_integer('classics', 101, 'The num of class')
FLAGS = flags.FLAGS
MOVING_AVERAGE_DECAY = 0.9999
model_save_dir = './models/model_100000_16_16_1e-1_0.9'

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def run_training():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.

    # Create model directory
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    use_pretrained_model = False
    model_filename = ''

    with tf.Graph().as_default():
        global_step = tf.get_variable(
                        'global_step',
                        [],
                        initializer=tf.constant_initializer(0),
                        trainable=False
                        )
        images_placeholder, labels_placeholder = placeholder_inputs(
                        FLAGS.batch_size * gpu_num,
                        FLAGS.num_frame_per_clib,
                        FLAGS.crop_size,
                        FLAGS.channels
                        )


        #opt_stable = tf.train.AdamOptimizer(1e-3)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=40000, decay_rate=0.1,staircase=True)
        opt_stable = tf.train.MomentumOptimizer(learning_rate, 0.9)
        i3d_tuple = InceptionI3d(
                                num_classes=FLAGS.classics,
                                spatial_squeeze=True,
                                final_endpoint='Logits',
                                name='inception_i3d'
                                )(images_placeholder, is_training=True)
        logit = i3d_tuple[0]
        loss = tower_loss(
                                logit,
                                labels_placeholder
                                )
        grads = opt_stable.compute_gradients(loss)
        accuracy = tower_acc(logit, labels_placeholder)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        apply_gradient_op = opt_stable.apply_gradients(grads, global_step=global_step)
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op, variables_averages_op)
        null_op = tf.no_op()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        # Create a session for running Ops on the Graph.
        sess = tf.Session(
                        config=tf.ConfigProto(allow_soft_placement=True)
                        )
        sess.run(init)
        # Create summary writter
        merged = tf.summary.merge_all()
    if os.path.isfile(model_filename) and use_pretrained_model:
        saver.restore(sess, model_filename)

    train_writer = tf.summary.FileWriter('./visual_logs/train_100000_16_16_1e-1_0.9', sess.graph)
    test_writer = tf.summary.FileWriter('./visual_logs/test_100000_16_16_1e-1_0.9', sess.graph)
    for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
                      filename='./list/ucf_list/train.list',
                      batch_size=FLAGS.batch_size * gpu_num,
                      num_frames_per_clip=FLAGS.num_frame_per_clib,
                      crop_size=FLAGS.crop_size,
                      shuffle=True
                      )
        sess.run(train_op, feed_dict={
                      images_placeholder: train_images,
                      labels_placeholder: train_labels
                      })
        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (step, duration))

        # Save a checkpoint and evaluate the model periodically.
        if (step) % 10 == 0 or (step + 1) == FLAGS.max_steps:
            print('Training Data Eval:')
            summary, acc, loss_all = sess.run(
                            [merged, accuracy, loss],
                            feed_dict={images_placeholder: train_images,
                                       labels_placeholder: train_labels
                                      })
            print("accuracy: " + "{:.5f}".format(acc))
            print("loss: " + "{:.5f}".format(loss_all))
            train_writer.add_summary(summary, step)
            print('Validation Data Eval:')
            val_images, val_labels, _, _, _ = input_data.read_clip_and_label(
                            filename='./list/ucf_list/test.list',
                            batch_size=FLAGS.batch_size * gpu_num,
                            num_frames_per_clip=FLAGS.num_frame_per_clib,
                            crop_size=FLAGS.crop_size,
                            shuffle=True
                            )
            summary, acc = sess.run(
                            [merged, accuracy],
                            feed_dict={
                                            images_placeholder: val_images,
                                            labels_placeholder: val_labels
                                            })
            print("accuracy: " + "{:.5f}".format(acc))
            test_writer.add_summary(summary, step)
        if (step) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            saver.save(sess, os.path.join(model_save_dir, 'c3d_ucf_model'), global_step=step)
    print("done")


def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
