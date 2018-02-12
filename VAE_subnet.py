
""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
slim = tf.contrib.slim
from utils import *
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
from data_loader import *

class generative_classifier(object):
    def __init__(self, config):

        self.mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

        self.is_vae = True

        # Parameters
        self.learning_rate = 0.001
        self.training_epochs = 200
        self.batch_size = 32
        self.display_step = 1
        self.examples_to_show = 10
        self.num_net = 10
        self.latent_dim = 8

        # Network Parameters
        self.imsize = 28# MNIST data input (img shape: 28*28)

        self.logdir, self.modeldir = creat_dir('VAE')
        copyfile('VAE_subnet.py', self.modeldir + '/' + 'DR_RNN.py')

        # tf Graph input (only pictures)
        self.X = tf.placeholder("float", [self.batch_size, self.imsize, self.imsize, 1])
        self.label = tf.placeholder("float", [self.batch_size, self.num_net])

    # Building the encoder
    def encoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        with tf.variable_scope('Enc'):
            h1 = slim.conv2d(x, 16, 3, 2, activation_fn=tf.nn.relu)
            h2 = slim.conv2d(h1, 32, 3, 2, activation_fn=tf.nn.relu)
            h3 = slim.conv2d(h2, 64, 3, 2, activation_fn=tf.nn.relu)
            h4 = slim.fully_connected(tf.reshape(h3,(self.batch_size,64*4*4)), 2*self.num_net*self.latent_dim, activation_fn=None)
        return h4

    # Building the decoder
    def decoder(self, sample_sub, label):

        out_sub = []
        for i, sample_i in enumerate(sample_sub):
            with tf.variable_scope('Dec{}'.format(i)):
                h1 = slim.fully_connected(sample_i, 64 * 4 * 4, activation_fn=tf.nn.relu)
                h2 = slim.conv2d_transpose(tf.reshape(h1, (self.batch_size, 4, 4, 64)), 64, 3, 2, activation_fn=tf.nn.relu)
                h3 = slim.conv2d_transpose(h2, 32, 3, 2, activation_fn=tf.nn.relu)
                h4 = slim.conv2d_transpose(h3, 16, 3, 2, activation_fn=tf.nn.relu)
                out = slim.conv2d(h4, 1, 5, 1, activation_fn=None, padding='valid')

                out_sub += [out]
                label_i = tf.reshape(label[:, i:i + 1], (self.batch_size, 1, 1, 1))
                out_sum = out*label_i if i==0 else out_sum+out*out_sum
                # out_sum = out
        return out_sum, out_sub

    def mysample(self, mean_var, num_net, is_vae):
        kld_sub = []
        sample_sub = []
        for i in range(num_net):
            mean_var_i = tf.slice(mean_var,(0,2*self.latent_dim*i),(self.batch_size,2*self.latent_dim))
            if is_vae:
                z_mean, z_log_var = tf.split(mean_var_i, 2, axis=1)
                batch_size, latent_dim = z_mean.get_shape().as_list()
                epsilon = tf.random_normal(shape=(batch_size, latent_dim), mean=0.,name='epsilon')
                sample_i = z_mean + tf.exp(z_log_var/ 2) * epsilon
                sample_sub += [sample_i]
                kld_i = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),axis = -1)
                kld_sub += [kld_i]
            else:
                sample_sub += [mean_var_i]
        return sample_sub, kld_sub

    def build_training_model(self):
        # Construct model
        with tf.variable_scope("Enc", reuse=False):
            self.mean_var = self.encoder(self.X)

        self.sample_sub, self.kld_sub = self.mysample(self.mean_var, self.num_net, is_vae=self.is_vae)

        with tf.variable_scope("Ded", reuse=False):
            self.out_sum, self.out_sub = self.decoder(self.sample_sub, self.label)
            self.rec = tf.sigmoid(self.out_sum)

        # Define loss and optimizer, minimize the squared error
        self.rec_cost = self.imsize*self.imsize*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X, logits=self.out_sum))
        if self.is_vae:
            self.kl_sum = tf.reduce_mean(list2tensor(self.kld_sub))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.rec_cost+self.kl_sum)
        else:
            self.kl_sum = tf.constant(0.)
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.rec_cost)



    def optimizing(self):
        # Launch the graph
        FLAGS = tf.app.flags.FLAGS
        tfconfig = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
        )
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        # Initializing the variables
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        total_batch = int(self.mnist.train.num_examples/self.batch_size)

        # Training cycle
        for epoch in range(self.training_epochs):
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = self.mnist.train.next_batch(self.batch_size)
                feed_dict = {self.X: batch_xs.reshape(self.batch_size, self.imsize, self.imsize, 1), self.label:batch_ys}#[:,0:1]
                _, rec_cost_val, kld_val = self.sess.run([self.optimizer, self.rec_cost, self.kl_sum], feed_dict)

            # Display logs per epoch step
            if epoch % self.display_step == 0:
                batch_xs, batch_ys = self.mnist.test.next_batch(self.batch_size)
                feed_dict = {self.X: batch_xs.reshape(self.batch_size, self.imsize, self.imsize, 1), self.label:batch_ys}
                _, test_rec_cost_val, test_kld_val = self.sess.run([self.optimizer, self.rec_cost, self.kl_sum], feed_dict)
                print("Epoch:", '%04d' % (epoch+1),
                      "rec_cost_end=", "{:.9f}".format(rec_cost_val),
                      "test_rec_cost_end=", "{:.9f}".format(test_rec_cost_val),
                      )

            if epoch % 50 == 0:
                self.saver = tf.train.Saver()
                snapshot_name = "%s_%s" % ('experiment', str(epoch))
                fn = self.saver.save(self.sess, "%s/%s.ckpt" % (self.modeldir, snapshot_name))
                print("Model saved in file: %s" % fn)

        print("Optimization Finished!")


    def build_testing_model(self, valid_x, valid_y, trained_model, para_list, name='mnist', gpu_idx=0, img_idx=None):
        loss_t, y_l, y_u, c_wd, theta, bias = para_list

        log_erri = 0
        log_idx = []
        log_x_input = []
        log_y_pred = []
        log_y_pred_ref = []
        log_rec_err = []
        log_rec_err_ref = []
        log_y_true = []

        with tf.device('gpu:{}'.format(gpu_idx)):
            # build network
            self.saver.restore(self.sess, trained_model)
            temp = set(tf.all_variables())
            self.x_input_opt_in = tf.placeholder(tf.float32, [1, 1, self.imsize, self.imsize], name='x_input_opt')
            self.y_input_opt_in = tf.placeholder(tf.float32, [1, len(valid_y[0])], name='y_input_opt')
            self.x_input_opt = self.x_input_opt_in
            self.y_input_opt = tf.Variable(self.y_input_opt_in)

            with tf.variable_scope("Dec", reuse=True):
                self.enc_z_opt = self.encoder(self.x_input_opt)
                self.z_opt = tf.Variable(self.enc_z_opt[0:1])
                self.z_opt_initial = tf.tile(self.z_opt, (2 * self.batch_size, 1))

                y_input_opt = tf.tile(self.y_input_opt, (self.batch_size, 1))
                self.out_sum_opt, self.out_sub_opt = self.decoder(self.z_opt_initial, self.y_input_opt)

                self.bias = tf.placeholder(tf.float32, [], name='bias')
                self.theta = tf.placeholder(tf.float32, [], name='theta')
                self.c_wd = tf.placeholder(tf.float32, [], name='c_wd')

                self.out_sum_opt_sig = tf.sigmoid(self.theta * self.out_sum_opt + self.bias)
                self.x_ref = tf.sigmoid(self.theta * self.x_input_opt + self.bias)

            self.loss_opt = tf.reduce_mean(tf.abs(self.out_sum_opt_sig - self.x_ref)) + self.c_wd * tf.norm(self.y_input_opt)
            opt = tf.train.AdamOptimizer(0.01)
            opt_sgd = tf.train.GradientDescentOptimizer(0.001)
            grads_g = opt_sgd.compute_gradients(self.loss_opt, var_list=[self.y_input_opt] + [self.z_opt])
            apply_gradient_opt = opt.apply_gradients(grads_g)

            def joint_optimization(x_input_opt):

                y0 = [0.5] * 10
                feed_dict = {self.c_wd: c_wd,
                             self.theta: theta,
                             self.bias: bias,
                             self.x_input_opt_in: np.expand_dims(x_input_opt, 0),
                             self.y_input_opt_in: np.expand_dims(y0, 0)}
                self.sess.run(tf.initialize_variables(set(tf.all_variables()) - temp), feed_dict)

                for i in range(50):
                    self.sess.run(apply_gradient_opt, feed_dict)

                y_pred = np.clip(self.y_input_opt.eval(session=self.sess), 0, 1.49)
                z_pred = self.sess.run(self.z_opt, feed_dict)
                rec_err = self.sess.run(tf.reduce_mean(tf.abs(self.out_sum_opt_sig - self.x_ref)), feed_dict)

                return y_pred, z_pred, rec_err

            def refine_classification(rec_err, y_pred, z_pred):
                loss_t = para_list[0][0]
                if rec_err > loss_t:
                    y_pred = [[0] * 10]
                    y_refine = [[0] * 10]
                    rec_err = [[0] * 10]
                    rec_err_ref = [[0] * 10]
                elif not y_pred.any():
                    # plt.imshow(self.sess.run(self.d_out_opt,
                    #                          {self.y_input_opt: self.y_input_opt.eval(session=self.sess),
                    #                           self.z0: self.z_pred})[
                    #                0, 0], interpolation='None')
                    # plt.savefig('idx{}_gen.png'.format(opt_idx))
                    # return self.y_pred, self.y_pred, self.y_pred, self.y_pred, rec_err, rec_err
                    y_pred = [[0] * 10]
                    y_refine = [[0] * 10]
                    rec_err = [[0] * 10]
                    rec_err_ref = [[0] * 10]
                else:
                    # consider some special cases:
                    mnist_idx = np.argsort(self.y_pred[0])
                    err_i = []
                    yy_tt = []
                    num = 3
                    for i in range(num):
                        yy = [0] * 10
                        yy[mnist_idx[-i - 1]] = 1
                        feed_dict = {self.c_wd: c_wd,
                                     self.theta: theta,
                                     self.bias: bias,
                                     self.x_input_opt_in: np.expand_dims(x_input_opt, 0),
                                     self.y_input_opt: [yy] * self.batch_size}

                        err_i += [np.mean(np.abs(self.sess.run(self.out_sum_opt_sig, feed_dict)[0, 0] - x_input_opt))]
                        yy_tt += [yy]
                    idx = np.argsort(np.asarray(err_i))
                    y_refine = yy_tt[idx[0]]

                    rec_err_refine = self.sess.run(tf.reduce_mean(tf.abs(self.out_sum_opt_sig- self.x_ref)), feed_dict)

                    # plt.imshow(self.sess.run(self.d_out_opt,
                    #                          {self.y_input_opt: self.y_pred, self.z0: self.z_pred})[
                    #                0, 0], interpolation='None')
                    # plt.savefig('idx{}_gen.png'.format(opt_idx))
                    z_pred_refine = 0.
                    return y_refine, z_pred_refine, rec_err_refine

            # optimization
            for opt_idx in range(len(valid_y)):
                if img_idx:
                    opt_idx = img_idx
                y_true = valid_y[opt_idx]
                x_input_opt = valid_x[opt_idx]
                # binarize first
                x_input_opt_round = np.round((x_input_opt - np.min(x_input_opt)) / (
                    np.max(x_input_opt) - np.min(x_input_opt)))
                x_input_opt = x_input_opt_round * (
                    np.max(x_input_opt) - np.min(x_input_opt)) + np.min(x_input_opt)

                self.y_pred, self.z_pred, self.rec_err = joint_optimization(x_input_opt)
                err_opt = y_true - np.round(self.y_pred)

                self.y_pred_refine, self.z_pred_refine, self.rec_err_refine = refine_classification(self.y_pred, self.z_pred, self.rec_err)
                # self.err_opt_refine = y_true - np.round(self.y_refine)

                log_idx += [opt_idx]
                log_y_true += [y_true]
                log_rec_err += [self.rec_err]
                log_y_pred += [self.y_pred]
                log_y_pred_ref += [self.y_pred_refine]
                log_rec_err_ref += [self.rec_err_refine]
                log_erri += int(np.asarray(self.rec_err_refine).any())

        # np.savez(self.modeldir+'/{}_{}_{}_{}_{}_{}_{}.npy'.format(name, 0.1, 0.3, 0.3, 0.001, 5, -2),
        #          log_err=log_erri, log_y_true=log_y_true, log_y_pred=log_y_pred, log_y_pred_ref=log_y_pred_ref,
        #          log_idx=log_idx)
        return log_erri

    def plotting(self):
        feed_dict = {
            self.X: self.mnist.test.images[:self.batch_size].reshape(self.batch_size, self.imsize, self.imsize, 1),
            self.label: self.mnist.test.labels[:self.batch_size]}
        # Applying encode and decode over test set
        encode_decode = self.sess.run(self.rec, feed_dict)
        f, a = plt.subplots(3, self.examples_to_show, figsize=(10, 2))
        for i in range(self.examples_to_show):
            a[0][i].imshow(np.reshape(self.mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
            # a[2][i].imshow(np.reshape(mid_encode_decode[i], (28, 28)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()


if __name__ == "__main__":
    cfg = {'delta_t': 1e-1,
           'time_start': 0,
           'time_end': 10,
           'num_y': 6,
           'num_layers': 5,
           'gamma': 0.01,
           'zeta': 0.9,
           'eps': 1e-8,
           'lr': 0.1,  # 0.2 for DR_RNN_1, 0.1 for DR_RNN_2 and 3, ??? for DR_RNN_4,
           'num_epochs': 15 * 10,
           'batch_size': 16,
           'data_fn': './data/3dof_sys_l.mat',  # './data/problem1.npz'
           }

    classifier = generative_classifier(cfg)
    classifier.build_training_model()
    classifier.optimizing()
    # classifier.build_testing_model()
    classifier.plotting()
    print('done')