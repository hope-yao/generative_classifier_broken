
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

slim = tf.contrib.slim
from source_code.utils import *
from tensorflow.examples.tutorials.mnist import input_data
from source_code.network_models import *
from source_code.plot_figures import *


class generative_classifier(object):
    def __init__(self):


        self.is_vae = True

        # Parameters
        self.learning_rate = 0.0003
        self.training_epochs = 200
        self.batch_size = 32
        self.display_step = 1
        self.examples_to_show = 10
        self.num_net = 10
        self.latent_dim = 8

        # Network Parameters
        self.imsize = 28# MNIST data input (img shape: 28*28)

        self.logdir, self.modeldir = creat_dir('VAE')
        copyfile('./source_code/VAE_subnet.py', self.modeldir + '/' + 'VAE_subnet.py')
        copyfile('./source_code/network_models.py', self.modeldir + '/' + 'network_models.py')
        copyfile('./source_code/plot_figures.py', self.modeldir + '/' + 'plot_figures.py')
        copyfile('./generate_attacks/generate_FGSM_attacks.py', self.modeldir + '/' + 'generate_FGSM_attacks.py')

        # tf Graph input (only pictures)
        self.X = tf.placeholder("float", [self.batch_size, self.imsize, self.imsize, 1])
        self.label = tf.placeholder("float", [self.batch_size, self.num_net])

        self.encoder = mnist28_encoder
        self.decoder = mnist28_decoder

        self.mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


    def build_training_model(self):
        # Construct model
        self.mean_var = self.encoder(self.X, self.num_net*self.latent_dim)

        self.sample_sub, self.kld_sub = mysample(self.mean_var, self.num_net, is_vae=self.is_vae)

        self.out_sum, self.out_sub = self.decoder(self.sample_sub, self.label)
        self.rec = tf.sigmoid(self.out_sum)

        # Define loss and optimizer, minimize the squared error
        self.rec_err = tf.reduce_mean(tf.abs(self.rec-self.X))
        self.rec_cost = self.imsize*self.imsize*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.X, logits=self.out_sum))
        if self.is_vae:
            self.kl_mean = tf.reduce_mean(list2tensor(self.kld_sub))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.rec_cost + self.kl_mean)
        else:
            self.kl_mean = tf.constant(0.)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.rec_cost)

        self.summary_writer = tf.summary.FileWriter(self.logdir)
        self.summary_op_train = tf.summary.merge([
            tf.summary.scalar("train/kl_mean_train", self.kl_mean),
            tf.summary.scalar("train/rec_cost_train", self.rec_cost),
            tf.summary.scalar("train/rec_err", self.rec_err),
            tf.summary.scalar("train/lr", self.learning_rate),
            # tf.summary.scalar("grad/grad1", grad1),
            # tf.summary.scalar("grad/grad2", grad2),
            # tf.summary.scalar("grad/grad3", grad3),
            # tf.summary.scalar("grad/grad4", grad4),
            # tf.summary.scalar("grad/grad5", grad5),
            # tf.summary.scalar("grad/grad6", grad6),
        ])

        self.summary_op_test = tf.summary.merge([
            tf.summary.scalar("test/kl_mean_test", self.kl_mean),
            tf.summary.scalar("test/rec_cost_test", self.rec_cost),
            tf.summary.scalar("test/rec_err", self.rec_err),
        ])



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
                feed_dict_train = {self.X: batch_xs.reshape(self.batch_size, self.imsize, self.imsize, 1), self.label:batch_ys}#[:,0:1]
                _, rec_cost_val, kld_val = self.sess.run([self.optimizer, self.rec_cost, self.kl_mean], feed_dict_train)

            # Display logs per epoch step
            if epoch % self.display_step == 0:
                batch_xs, batch_ys = self.mnist.test.next_batch(self.batch_size)
                feed_dict_test = {self.X: batch_xs.reshape(self.batch_size, self.imsize, self.imsize, 1), self.label:batch_ys}
                _, test_rec_cost_val, test_kld_val = self.sess.run([self.optimizer, self.rec_cost, self.kl_mean], feed_dict_test)
                print("Epoch:", '%04d' % (epoch+1),
                      "rec_cost_end=", "{:.9f}".format(rec_cost_val),
                      "test_rec_cost_end=", "{:.9f}".format(test_rec_cost_val),
                      )
                summary_train = self.sess.run(self.summary_op_train, feed_dict_train)
                summary_test = self.sess.run(self.summary_op_test, feed_dict_test)
                self.summary_writer.add_summary(summary_train, epoch)
                self.summary_writer.add_summary(summary_test, epoch)
                self.summary_writer.flush()

            if epoch % 50 == 0:
                self.saver = tf.train.Saver()
                snapshot_name = "%s_%s" % ('experiment', str(epoch))
                fn = self.saver.save(self.sess, "%s/%s.ckpt" % (self.modeldir, snapshot_name))
                print("Model saved in file: %s" % fn)

        print("Optimization Finished!")
    #
    #
    # def testing_model(self, valid_x, valid_y, trained_model_ckpt, para_list, name='mnist', gpu_idx=0, img_idx=None):
    #     loss_t, y_l, y_u, c_wd, theta, bias = para_list
    #
    #     log_erri = 0
    #     log_idx = []
    #     log_x_input = []
    #     log_y_pred = []
    #     log_y_pred_ref = []
    #     log_rec_err = []
    #     log_rec_err_ref = []
    #     log_y_true = []
    #
    #     with tf.device('gpu:{}'.format(gpu_idx)):
    #         # build network
    #         self.saver.restore(self.sess, trained_model_ckpt)
    #         temp = set(tf.all_variables())
    #         self.x_input_opt_in = tf.placeholder(tf.float32, [1, 1, self.imsize, self.imsize], name='x_input_opt')
    #         self.y_input_opt_in = tf.placeholder(tf.float32, [1, len(valid_y[0])], name='y_input_opt')
    #         self.x_input_opt = self.x_input_opt_in
    #         self.y_input_opt = tf.Variable(self.y_input_opt_in)
    #
    #         with tf.variable_scope("Dec", reuse=True):
    #             self.enc_z_opt = self.encoder(self.x_input_opt)
    #             self.z_opt = tf.Variable(self.enc_z_opt[0:1])
    #             self.z_opt_initial = tf.tile(self.z_opt, (2 * self.batch_size, 1))
    #
    #             y_input_opt = tf.tile(self.y_input_opt, (self.batch_size, 1))
    #             self.out_sum_opt, self.out_sub_opt = self.decoder(self.z_opt_initial, self.y_input_opt)
    #
    #             self.bias = tf.placeholder(tf.float32, [], name='bias')
    #             self.theta = tf.placeholder(tf.float32, [], name='theta')
    #             self.c_wd = tf.placeholder(tf.float32, [], name='c_wd')
    #
    #             self.out_sum_opt_sig = tf.sigmoid(self.theta * self.out_sum_opt + self.bias)
    #             self.x_ref = tf.sigmoid(self.theta * self.x_input_opt + self.bias)
    #
    #         self.loss_opt = tf.reduce_mean(tf.abs(self.out_sum_opt_sig - self.x_ref)) + self.c_wd * tf.norm(self.y_input_opt)
    #         opt = tf.train.AdamOptimizer(0.01)
    #         opt_sgd = tf.train.GradientDescentOptimizer(0.001)
    #         grads_g = opt_sgd.compute_gradients(self.loss_opt, var_list=[self.y_input_opt] + [self.z_opt])
    #         apply_gradient_opt = opt.apply_gradients(grads_g)
    #
    #         def joint_optimization(x_input_opt):
    #
    #             y0 = [0.5] * 10
    #             feed_dict = {self.c_wd: c_wd,
    #                          self.theta: theta,
    #                          self.bias: bias,
    #                          self.x_input_opt_in: np.expand_dims(x_input_opt, 0),
    #                          self.y_input_opt_in: np.expand_dims(y0, 0)}
    #             self.sess.run(tf.initialize_variables(set(tf.all_variables()) - temp), feed_dict)
    #
    #             for i in range(50):
    #                 self.sess.run(apply_gradient_opt, feed_dict)
    #
    #             y_pred = np.clip(self.y_input_opt.eval(session=self.sess), 0, 1.49)
    #             z_pred = self.sess.run(self.z_opt, feed_dict)
    #             rec_err = self.sess.run(tf.reduce_mean(tf.abs(self.out_sum_opt_sig - self.x_ref)), feed_dict)
    #
    #             return y_pred, z_pred, rec_err
    #
    #         def refine_classification(rec_err, y_pred, z_pred):
    #             loss_t = para_list[0][0]
    #             if rec_err > loss_t:
    #                 y_pred = [[0] * 10]
    #                 y_refine = [[0] * 10]
    #                 rec_err = [[0] * 10]
    #                 rec_err_ref = [[0] * 10]
    #             elif not y_pred.any():
    #                 # plt.imshow(self.sess.run(self.d_out_opt,
    #                 #                          {self.y_input_opt: self.y_input_opt.eval(session=self.sess),
    #                 #                           self.z0: self.z_pred})[
    #                 #                0, 0], interpolation='None')
    #                 # plt.savefig('idx{}_gen.png'.format(opt_idx))
    #                 # return self.y_pred, self.y_pred, self.y_pred, self.y_pred, rec_err, rec_err
    #                 y_pred = [[0] * 10]
    #                 y_refine = [[0] * 10]
    #                 rec_err = [[0] * 10]
    #                 rec_err_ref = [[0] * 10]
    #             else:
    #                 # consider some special cases:
    #                 mnist_idx = np.argsort(self.y_pred[0])
    #                 err_i = []
    #                 yy_tt = []
    #                 num = 3
    #                 for i in range(num):
    #                     yy = [0] * 10
    #                     yy[mnist_idx[-i - 1]] = 1
    #                     feed_dict = {self.c_wd: c_wd,
    #                                  self.theta: theta,
    #                                  self.bias: bias,
    #                                  self.x_input_opt_in: np.expand_dims(x_input_opt, 0),
    #                                  self.y_input_opt: [yy] * self.batch_size}
    #
    #                     err_i += [np.mean(np.abs(self.sess.run(self.out_sum_opt_sig, feed_dict)[0, 0] - x_input_opt))]
    #                     yy_tt += [yy]
    #                 idx = np.argsort(np.asarray(err_i))
    #                 y_refine = yy_tt[idx[0]]
    #
    #                 rec_err_refine = self.sess.run(tf.reduce_mean(tf.abs(self.out_sum_opt_sig- self.x_ref)), feed_dict)
    #
    #                 # plt.imshow(self.sess.run(self.d_out_opt,
    #                 #                          {self.y_input_opt: self.y_pred, self.z0: self.z_pred})[
    #                 #                0, 0], interpolation='None')
    #                 # plt.savefig('idx{}_gen.png'.format(opt_idx))
    #                 z_pred_refine = 0.
    #                 return y_refine, z_pred_refine, rec_err_refine
    #
    #         # optimization
    #         for opt_idx in range(len(valid_y)):
    #             if img_idx:
    #                 opt_idx = img_idx
    #             y_true = valid_y[opt_idx]
    #             x_input_opt = valid_x[opt_idx]
    #             # binarize first
    #             x_input_opt_round = np.round((x_input_opt - np.min(x_input_opt)) / (
    #                 np.max(x_input_opt) - np.min(x_input_opt)))
    #             x_input_opt = x_input_opt_round * (
    #                 np.max(x_input_opt) - np.min(x_input_opt)) + np.min(x_input_opt)
    #
    #             self.y_pred, self.z_pred, self.rec_err = joint_optimization(x_input_opt)
    #             err_opt = y_true - np.round(self.y_pred)
    #
    #             self.y_pred_refine, self.z_pred_refine, self.rec_err_refine = refine_classification(self.y_pred, self.z_pred, self.rec_err)
    #             # self.err_opt_refine = y_true - np.round(self.y_refine)
    #
    #             log_idx += [opt_idx]
    #             log_y_true += [y_true]
    #             log_rec_err += [self.rec_err]
    #             log_y_pred += [self.y_pred]
    #             log_y_pred_ref += [self.y_pred_refine]
    #             log_rec_err_ref += [self.rec_err_refine]
    #             log_erri += int(np.asarray(self.rec_err_refine).any())
    #
    #     # np.savez(self.modeldir+'/{}_{}_{}_{}_{}_{}_{}.npy'.format(name, 0.1, 0.3, 0.3, 0.001, 5, -2),
    #     #          log_err=log_erri, log_y_true=log_y_true, log_y_pred=log_y_pred, log_y_pred_ref=log_y_pred_ref,
    #     #          log_idx=log_idx)
    #     return log_erri
    #

    #
    # def plotting(self, figure_path):
    #     feed_dict = {
    #         self.X: self.mnist.test.images[:self.batch_size].reshape(self.batch_size,
    #                                                                  self.imsize,
    #                                                                  self.imsize, 1),
    #         self.label: self.mnist.test.labels[:self.batch_size]}
    #     # Applying encode and decode over test set
    #     encode_decode = self.sess.run(self.rec, feed_dict)
    #     f, a = plt.subplots(3, self.examples_to_show, figsize=(10, 2))
    #     for i in range(self.examples_to_show):
    #         a[0][i].imshow(np.reshape(self.mnist.test.images[i], (28, 28)))
    #         a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    #         # a[2][i].imshow(np.reshape(mid_encode_decode[i], (28, 28)))
    #     f.show()
    #     f.savefig(figure_path)

if __name__ == "__main__":

    classifier = generative_classifier()
    classifier.build_training_model()
    classifier.optimizing()
    plotting(classifier, classifier.modeldir+'/test.png')
    print('done')
