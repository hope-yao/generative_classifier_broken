
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
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 200
batch_size = 32
display_step = 1
examples_to_show = 10
num_net = 10
latent_dim = 8

# Network Parameters
n_hidden_1 = 128 # 1st layer num features
n_hidden_3 = 32# 3nd layer num features
n_hidden_4 = latent_dim# 4nd layer num features
imsize = 28# MNIST data input (img shape: 28*28)

logdir, modeldir = creat_dir('VAE')
mid_flag = 0

# tf Graph input (only pictures)
X = tf.placeholder("float", [batch_size, imsize, imsize, 1])
label = tf.placeholder("float", [batch_size, num_net])

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    with tf.variable_scope('Enc'):
        h1 = slim.conv2d(x, 16, 3, 2, activation_fn=tf.nn.relu)
        h2 = slim.conv2d(h1, 32, 3, 2, activation_fn=tf.nn.relu)
        h3 = slim.conv2d(h2, 64, 3, 2, activation_fn=tf.nn.relu)
        h4 = slim.fully_connected(tf.reshape(h3,(batch_size,64*4*4)), 2*num_net*latent_dim, activation_fn=None)
    return h4

# Building the decoder
def decoder(num_net, label, mean_var_end):

    out_sub = []
    kld_end_sub = []
    for i in range(num_net):
        with tf.variable_scope('Dec{}'.format(i)):
            mean_var_end_i = tf.slice(mean_var_end,(0,2*latent_dim*i),(batch_size,2*latent_dim))
            sample_end_i, kld_end_i = mysample(mean_var_end_i)

            # mean_var_mid_i = tf.slice(mean_var_mid,(0,2*latent_dim*i),(batch_size,2*latent_dim))
            # sample_mid_i, kld_mid_i = mysample(mean_var_mid_i)

            h1 = slim.fully_connected(mean_var_end_i, 64 * 4 * 4, activation_fn=tf.nn.relu)
            h2 = slim.conv2d_transpose(tf.reshape(h1, (batch_size, 4, 4, 64)), 64, 3, 2, activation_fn=tf.nn.relu)
            h3 = slim.conv2d_transpose(h2, 32, 3, 2, activation_fn=tf.nn.relu)
            h4 = slim.conv2d_transpose(h3, 16, 3, 2, activation_fn=tf.nn.relu)
            out = slim.conv2d(h4, 1, 5, 1, activation_fn=None, padding='valid')

            out_sub += [out]
            # label_i = tf.reshape(label[:, i:i + 1], (batch_size, 1, 1, 1))
            # out_sum = out*label_i if i==0 else out_sum+out*out_sum
            out_sum = out
            kld_end_sub += [kld_end_i]
            kld_end_sum = kld_end_i if i == 0 else kld_end_sum + kld_end_i
    return out_sum, out_sub, kld_end_sub, kld_end_sum

def mysample(mean_var):
    z_mean, z_log_var = tf.split(mean_var, 2, axis=1)
    batch_size, latent_dim = z_mean.get_shape().as_list()
    epsilon = tf.random_normal(shape=(batch_size, latent_dim), mean=0.,name='epsilon')
    sample = z_mean + tf.exp(z_log_var/ 2) * epsilon
    kld = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),axis = -1)
    return sample, kld


# Construct model
mean_var_end = encoder(X)
out_sum, out_sub, kld_end_sub, kld_end_sum = decoder(num_net, label, mean_var_end)

# Prediction
y_pred = out_sum
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
# rec_cost_end = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
rec_cost_end = imsize*imsize*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))

optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(rec_cost_end)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph

FLAGS = tf.app.flags.FLAGS
tfconfig = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True,
)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
sess.run(init)
total_batch = int(mnist.train.num_examples/batch_size)
# Training cycle
for epoch in range(training_epochs):
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, rec_cost_end_val, kld_end_val\
            = sess.run([optimizer, rec_cost_end, kld_end_sum], feed_dict={X: batch_xs.reshape(batch_size,imsize,imsize,1)})#, label:batch_ys})
    # Display logs per epoch step
    if epoch % display_step == 0:
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        _, test_rec_cost_end_val, test_kld_end_val\
            = sess.run([optimizer, rec_cost_end, kld_end_sum], feed_dict={X: batch_xs.reshape(batch_size,imsize,imsize,1)})#, label:batch_ys})
        print("Epoch:", '%04d' % (epoch+1),
              "rec_cost_end=", "{:.9f}".format(rec_cost_end_val),
              # "kld_end=", "{:.9f}".format(kld_end_val),
              "test_rec_cost_end=", "{:.9f}".format(test_rec_cost_end_val),
              # "test_kld_end=", "{:.9f}".format(test_kld_end_val),

              # "rec_cost_mid=", "{:.9f}".format(rec_cost_mid_val),
              # "kld_mid=", "{:.9f}".format(kld_mid_val),
              # "test_rec_cost_mid=", "{:.9f}".format(test_rec_cost_mid_val),
              # "test_kld_mid=", "{:.9f}".format(test_kld_mid_val),
              )

    if epoch % 50 == 0:
        saver = tf.train.Saver()
        snapshot_name = "%s_%s" % ('experiment', str(epoch))
        fn = saver.save(sess, "%s/%s.ckpt" % (modeldir, snapshot_name))
        print("Model saved in file: %s" % fn)

print("Optimization Finished!")

# Applying encode and decode over test set
encode_decode = sess.run(
    tf.sigmoid(y_pred), feed_dict={X: mnist.test.images[:batch_size].reshape(batch_size,imsize,imsize,1)}) #, label:mnist.test.labels[:batch_size]
# mid_encode_decode = sess.run(
#     rec_mid, feed_dict={X: mnist.test.images[:batch_size], label:mnist.test.labels[:batch_size]})
# Compare original images with their reconstructions
f, a = plt.subplots(3, examples_to_show, figsize=(10, 2))
for i in range(examples_to_show):
    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    # a[2][i].imshow(np.reshape(mid_encode_decode[i], (28, 28)))
f.show()
plt.draw()
plt.waitforbuttonpress()