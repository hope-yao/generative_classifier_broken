# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Contains a variant of the LeNet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.utils import *
slim = tf.contrib.slim

class Lenet():
    def __init__(self):
        self.batch_size = 32
        self.imsize = 28
        self.channels = 1
        self.num_class = 10
        self.learning_rate = 0.01
        self.mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        self.training_epochs = 200
        self.logdir, self.modeldir = creat_dir('LENET')

        self.input = tf.placeholder(tf.float32, [self.batch_size, self.imsize, self.imsize, self.channels])
        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.num_class])
        # self.input = tf.Variable(self.input_pl)

    def add_layers(self, images, num_classes=10, is_training=False,
              dropout_keep_prob=0.5,
              prediction_fn=slim.softmax,
              scope='LeNet'):
        """Creates a variant of the LeNet model.

        Note that since the output is a set of 'logits', the values fall in the
        interval of (-infinity, infinity). Consequently, to convert the outputs to a
        probability distribution over the characters, one will need to convert them
        using the softmax function:

            logits = lenet.lenet(images, is_training=False)
            probabilities = tf.nn.softmax(logits)
            predictions = tf.argmax(logits, 1)

        Args:
        images: A batch of `Tensors` of size [batch_size, height, width, channels].
        num_classes: the number of classes in the dataset.
        is_training: specifies whether or not we're currently training the model.
          This variable will determine the behaviour of the dropout layer.
        dropout_keep_prob: the percentage of activation values that are retained.
        prediction_fn: a function to get predictions out of logits.
        scope: Optional variable_scope.

        Returns:
        logits: the pre-softmax activations, a tensor of size
          [batch_size, `num_classes`]
        end_points: a dictionary from components of the network to the corresponding
          activation.
        """
        end_points = {}

        # with tf.variable_scope(scope, 'LeNet', [images, num_classes]):
        net = slim.conv2d(images, 32, 5, scope='conv1')
        net = slim.max_pool2d(net, 2, 2, scope='pool1')
        net = slim.conv2d(net, 64, 5, scope='conv2')
        net = slim.max_pool2d(net, 2, 2, scope='pool2')
        net = slim.flatten(net)
        end_points['Flatten'] = net

        mid_output = net = slim.fully_connected(net, 1024, scope='fc3')
        # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
        #                    scope='dropout3')
        logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                      scope='fc4')

        end_points['mid_output'] = mid_output
        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

        return end_points

    def build_model(self):

        with tf.variable_scope('LENET', reuse=False) as lenet_var:
            self.end_points = self.add_layers(self.input, num_classes=10, is_training=True,
                           dropout_keep_prob=0.99)
            self.lenet_var = tf.contrib.framework.get_variables(lenet_var)

        # Define loss and optimizer, minimize the squared error
        self.rec_err = tf.reduce_mean(tf.abs(self.end_points['Predictions'] - self.labels))
        self.rec_cost = self.imsize * self.imsize * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.end_points['Logits']))

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_g = optimizer.compute_gradients(self.rec_cost, var_list=self.lenet_var)
        self.apply_gradient_training = optimizer.apply_gradients(grads_g)

    def train_model(self):
        # Launch the graph
        FLAGS = tf.app.flags.FLAGS
        tfconfig = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        # Initializing the variables
        self.sess.run(tf.initialize_variables(set(tf.all_variables())))#-set([self.input])))

        total_batch = int(self.mnist.train.num_examples/self.batch_size)
        # Training cycle
        for epoch in range(self.training_epochs):
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = self.mnist.train.next_batch(self.batch_size)
                feed_dict_train = {self.input: batch_xs.reshape(self.batch_size, self.imsize, self.imsize, 1), self.labels:batch_ys}#[:,0:1]
                _, rec_cost_val, rec_err_val = self.sess.run([self.apply_gradient_training, self.rec_cost, self.rec_err], feed_dict_train)

            # Display logs per epoch step
            batch_xs, batch_ys = self.mnist.test.next_batch(self.batch_size)
            feed_dict_test = {self.input: batch_xs.reshape(self.batch_size, self.imsize, self.imsize, 1), self.labels:batch_ys}
            test_rec_cost_val, test_rec_err_val = self.sess.run([self.rec_cost, self.rec_err], feed_dict_test)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_rec_cost=", "{:.9f}".format(rec_cost_val),
                  "test_rec_cost=", "{:.9f}".format(test_rec_cost_val),
                  "train_rec_err=", "{:.9f}".format(rec_err_val),
                  "test_rec_err=", "{:.9f}".format(test_rec_err_val),
                  )

            if epoch % 50 == 0:
                self.saver = tf.train.Saver()
                snapshot_name = "%s_%s" % ('experiment', str(epoch))
                fn = self.saver.save(self.sess, "%s/%s.ckpt" % (self.modeldir, snapshot_name))
                print("Model saved in file: %s" % fn)

        print("Optimization Finished!")



if __name__ == '__main__':
    lenet = Lenet()
    lenet.build_model()
    # lenet.train_model()

    from generate_attacks.generate_FGSM_attacks import get_attack_directions
    # modle_path = 'models/LENET/LENET_2018_02_13_15_49_34/experiment_19.ckpt'

    # Launch the graph
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)
    trained_model_ckpt = '/home/exx/Documents/Hope/generative_classifier/models/LENET/LENET_2018_02_13_17_27_15/experiment_0.ckpt'
    saver = tf.train.Saver()
    saver.restore(sess, trained_model_ckpt)

    get_attack_directions(lenet, sess)

