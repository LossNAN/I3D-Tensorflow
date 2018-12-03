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
import sys
sys.path.append('../../')
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import math
import numpy as np
from i3d_nonlocal import InceptionI3d
from i3d_utils import *
from tensorflow.python import pywrap_tensorflow

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 4
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 8, 'Batch size.')
flags.DEFINE_integer('num_frame_per_clib', 64, 'Nummber of frames per clib')
flags.DEFINE_integer('sample_rate', 4, 'Sample rate for clib')
flags.DEFINE_integer('crop_size', 224, 'Crop_size')
flags.DEFINE_integer('classics', 101, 'The num of class')
flags.DEFINE_integer('block_num', 0, 'The num of nonlocal block')
flags.DEFINE_float('weight_decay', 0.000001, 'weight decay')
FLAGS = flags.FLAGS
model_save_dir = './models/%dGPU_sgd%dblock_scratch_400000_8_64_0.0001_decay'%(gpu_num, FLAGS.block_num)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def run_training():
    # Get the sets of images and labels for training, validation, and
    # Tell TensorFlow that the model will be built into the default Graph.

    # Create model directory
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    rgb_pre_model_save_dir = "/home/project/I3D/I3D/checkpoints/rgb_imagenet"

    video_path_list = np.load('./data_list/train_data_list.npy')
    label_list = np.load('./data_list/train_label_list.npy')
    with tf.Graph().as_default():
        global_step = tf.get_variable(
            'global_step',
            [],
            dtype=tf.int32,
            initializer=tf.constant_initializer(0),
            trainable=False
        )
        train_input_queue = tf.train.slice_input_producer([video_path_list, label_list], shuffle=True)
        video_path = train_input_queue[0]
        train_label = train_input_queue[1]

        rgb_train_images, _, _ = tf.py_func(func=input_data.get_frames,
                                            inp=[video_path, -1, FLAGS.num_frame_per_clib, FLAGS.crop_size,
                                                 FLAGS.sample_rate, False],
                                            Tout=[tf.float32, tf.double, tf.int64],
                                            )

        batch_videos, batch_labels = tf.train.batch([rgb_train_images, train_label],
                                                    batch_size=FLAGS.batch_size * gpu_num, capacity=200,
                                                    num_threads=20, shapes=[(FLAGS.num_frame_per_clib / FLAGS.sample_rate, FLAGS.crop_size, FLAGS.crop_size, 3), ()])
        opt_rgb = tf.train.AdamOptimizer(learning_rate)
        #opt_nonlocal = tf.train.AdamOptimizer(learning_rate*10)
        #opt_rgb = tf.train.MomentumOptimizer(learning_rate, 0.9)
        #opt_rgb = tf.train.GradientDescentOptimizer(learning_rate)
        tower_grads = []
        logits = []
        loss = []
        with tf.variable_scope(tf.get_variable_scope()):
            for gpu_index in range(0, gpu_num):
                with tf.device('/gpu:%d' % gpu_index):
                    with tf.name_scope('GPU_%d' % gpu_index):
                        rgb_logit, _ = InceptionI3d(
                                                num_classes=FLAGS.classics,
                                                spatial_squeeze=True,
                                                final_endpoint='Logits',
                                                block_num=FLAGS.block_num
                                                )(batch_videos[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size,:,:,:,:], True)
                        rgb_loss = tower_loss(
                                                    rgb_logit,
                                                    batch_labels[gpu_index * FLAGS.batch_size:(gpu_index + 1) * FLAGS.batch_size],
                                                    FLAGS.weight_decay
                                                    )
                        tf.get_variable_scope().reuse_variables()
                        rgb_grads = opt_rgb.compute_gradients(rgb_loss)
                tower_grads.append(rgb_grads)
                logits.append(rgb_logit)
                loss.append(rgb_loss)
        logits = tf.concat(logits, 0)
        accuracy = tower_acc(logits, batch_labels)
        grads = average_gradients(tower_grads)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        rgb_variable_map = {}
        i3d_map = {}
        nonlocal_map = {}
        for variable in tf.global_variables():
            if 'NonLocalBlock' in variable.name:
                nonlocal_map[variable.name] = variable
            else:
                i3d_map[variable.name] = variable

            if variable.name.split('/')[0] == 'RGB' and \
                    'Adam' not in variable.name.split('/')[-1] and \
                    'NonLocal' not in variable.name:
                #rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
                rgb_variable_map[variable.name.replace(':0', '')] = variable

        with tf.control_dependencies(update_ops):
            apply_gradient_rgb = opt_rgb.apply_gradients(grads, global_step=global_step)
            if FLAGS.block_num >= 0:
                train_op = tf.group(apply_gradient_rgb)
            else:
                nonlocal_grads = opt_nonlocal.compute_gradients(rgb_loss, var_list=nonlocal_map)
                apply_gradient_nonlocal = opt_nonlocal.apply_gradients(nonlocal_grads, global_step=global_step)
                train_op = tf.group(apply_gradient_rgb, apply_gradient_nonlocal)
            null_op = tf.no_op()

        # Create a session for running Ops on the Graph.
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sess.run(init)
        # Create summary writter
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('rgb_loss', tf.reduce_mean(loss))
        tf.summary.scalar('learning_rate', learning_rate)
        merged = tf.summary.merge_all()
    # load pre_train models

    ckpt = tf.train.get_checkpoint_state(rgb_pre_model_save_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("loading checkpoint %s,waiting......" % ckpt.model_checkpoint_path)
        rgb_saver.restore(sess, ckpt.model_checkpoint_path)
        print("load complete!")

    train_writer = tf.summary.FileWriter('./visual_logs/%dGPU_sgd%dblock_train_scratch_400000_8_64_0.0001_decay'%(gpu_num, FLAGS.block_num), sess.graph)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    for step in range(FLAGS.max_steps):
        start_time = time.time()
        sess.run(train_op)
        duration = time.time() - start_time
        print('Step %d: %.3f sec, end time : after %.3f days' % (
        step, duration, (FLAGS.max_steps - step) * duration / 86400))

        if step % 10 == 0 or (step + 1) == FLAGS.max_steps:
            print('Training Data Eval:')
            summary, acc, loss_rgb = sess.run([merged, accuracy, loss])
            print("accuracy: " + "{:.5f}".format(acc))
            print("rgb_loss: " + "{:.5f}".format(np.mean(loss_rgb)))
            train_writer.add_summary(summary, step)

        if (step + 1) % 2000 == 0 or (step + 1) == FLAGS.max_steps:
            saver.save(sess, os.path.join(model_save_dir, 'model'), global_step=step)

    coord.request_stop()
    coord.join(threads)
    print("done")

def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
