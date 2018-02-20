from VAE_subnet import *


def build_classify_model(model):
    temp = set(tf.all_variables())
    model.x_input_opt_in = tf.placeholder(tf.float32, [1, model.imsize, model.imsize, 1], name='x_input_opt')
    model.y_input_opt_in = tf.placeholder(tf.float32, [1, model.num_net], name='y_input_opt')
    model.x_input_opt = tf.tile(model.x_input_opt_in, (model.batch_size, 1, 1, 1))
    model.y_input_opt = tf.Variable(model.y_input_opt_in)

    enc_z = model.encoder(model.x_input_opt, model.latent_dim * model.num_net, reuse=True)

    model.enc_z_sample, _ = mysample(enc_z[0:1], model.num_net, is_vae=True)  # just to split z value

    model.z_initial = tf.Variable(model.enc_z_sample)
    z_opt = []
    for i in range(model.num_net):
        z_opt += [tf.tile(model.z_initial[i], (model.batch_size, 1))]
    y_input_opt = tf.tile(model.y_input_opt, (model.batch_size, 1))
    model.out_sum_opt, model.out_sub_opt = model.decoder(z_opt, y_input_opt, reuse=True)
    model.gen_img = tf.sigmoid(model.out_sum_opt)

    model.bias = tf.placeholder(tf.float32, [], name='bias')
    model.theta = tf.placeholder(tf.float32, [], name='theta')
    model.c_wd = tf.placeholder(tf.float32, [], name='c_wd')

    model.gen_img_sig = tf.sigmoid(model.theta * model.gen_img + model.bias)
    model.x_ref = tf.sigmoid(model.theta * model.x_input_opt + model.bias)

    model.loss_opt = tf.reduce_mean(tf.abs(model.gen_img_sig - model.x_ref)) \
                     + model.c_wd * tf.norm(model.y_input_opt, ord=1)
    opt = tf.train.AdamOptimizer(0.01)
    # opt_sgd = tf.train.GradientDescentOptimizer(0.001)
    grads_g = opt.compute_gradients(model.loss_opt, var_list=[model.y_input_opt, model.z_initial])
    apply_gradient_op = opt.apply_gradients(grads_g)

    init_op = tf.initialize_variables(set(tf.all_variables()) - temp)
    return apply_gradient_op, init_op


def joint_optimization(model, sess, para_list, x_input_opt, apply_gradient_op, init_op):
    loss_t, y_l, y_u, c_wd, theta, bias = para_list

    y0 = [0.5] * 10
    feed_dict = {model.c_wd: c_wd,
                 model.theta: theta,
                 model.bias: bias,
                 model.x_input_opt_in: np.expand_dims(x_input_opt, 0),
                 model.y_input_opt_in: np.expand_dims(y0, 0)}
    sess.run(init_op, feed_dict)

    y_pred_hist = []
    for i in range(50):
        _, y_pred = sess.run([apply_gradient_op, model.y_input_opt], feed_dict)
        y_pred_hist += [y_pred]

    # y_pred = np.clip(model.y_input_opt.eval(session=sess), 0, 1.49)
    z_pred = sess.run(model.z_initial, feed_dict)
    rec_err = sess.run(tf.reduce_mean(tf.abs(model.gen_img_sig - model.x_ref)), feed_dict)

    return y_pred_hist, z_pred, rec_err


def refine_classification(model, sess, x_input_opt, para_list, rec_err, y_pred, z_pred):
    loss_t, y_l, y_u, c_wd, theta, bias = para_list
    if rec_err > loss_t:
        y_refine = -1. * np.ones_like(y_pred)
        z_pred_refine = -1. * np.ones_like(z_pred)
        rec_err_refine = -1.
    elif not y_pred.any():
        y_refine = -1. * np.ones_like(y_pred)
        z_pred_refine = -1. * np.ones_like(z_pred)
        rec_err_refine = -1.
    else:
        # consider some special cases:
        mnist_idx = np.argsort(y_pred[0])
        err_i = []
        yy_tt = []
        num = 3
        for i in range(num):
            yy = [0] * 10
            yy[mnist_idx[-i - 1]] = 1
            feed_dict = {model.c_wd: c_wd,
                         model.theta: theta,
                         model.bias: bias,
                         model.x_input_opt_in: np.expand_dims(x_input_opt, 0),
                         model.y_input_opt: [yy]}

            err_i += [np.mean(np.abs(sess.run(model.gen_img_sig- model.x_ref, feed_dict)))]
            yy_tt += [yy]
        idx = np.argsort(np.asarray(err_i))
        y_refine = yy_tt[idx[0]]

        rec_err_refine = sess.run(tf.reduce_mean(tf.abs(model.gen_img_sig - model.x_ref)), feed_dict)

        # plt.imshow(self.sess.run(self.d_out_opt,
        #                          {self.y_input_opt: self.y_pred, self.z0: self.z_pred})[
        #                0, 0], interpolation='None')
        # plt.savefig('idx{}_gen.png'.format(opt_idx))
        z_pred_refine = z_pred
    return y_refine, z_pred_refine, rec_err_refine


def testing_classify_model(sess, model, apply_gradient_op, init_op, valid_x, valid_y, para_list, fn=None, img_idx=None):
    out_dict = {}

    # optimization
    out_dict['log_err'] = []
    out_dict['log_idx'] = []
    out_dict['log_y_pred'] = []
    out_dict['log_y_pred_ref'] = []
    out_dict['log_rec_err'] = []
    out_dict['log_rec_err_ref'] = []
    out_dict['log_y_true'] = []

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

        # perform joint optimization
        y_pred_hist, z_pred, rec_err = joint_optimization(model, sess, para_list, x_input_opt, apply_gradient_op, init_op)
        y_pred = y_pred_hist[-1]
        # refine classification
        y_pred_refine, z_pred_refine, rec_err_refine = \
            refine_classification(model, sess, x_input_opt, para_list, rec_err, y_pred, z_pred)

        out_dict['log_idx'] += [opt_idx]
        out_dict['log_y_true'] += [y_true]
        out_dict['log_rec_err'] += [rec_err]
        out_dict['log_y_pred'] += [y_pred]
        out_dict['log_y_pred_ref'] += [y_pred_refine]
        out_dict['log_rec_err_ref'] += [rec_err_refine]
        if int((np.asarray(y_pred_refine)-y_true).any()):
            out_dict['log_err'] += [opt_idx]
            # save wrong cases
            img = sess.run(model.gen_img, {model.y_input_opt: y_pred, model.z_initial: z_pred})
            plt.imshow(np.squeeze(img[0]), interpolation='None')
            plt.title('prediction: {}'.format(np.argmax(y_pred_refine))) #
            plt.savefig(model.modeldir + '/idx{}_gen.png'.format(opt_idx))
            plt.imshow(np.squeeze(valid_x[opt_idx]), interpolation='None')
            plt.title('rec_err: {}'.format(rec_err)) #prediction: {}
            plt.savefig(model.modeldir + '/idx{}_input.png'.format(opt_idx))

        if opt_idx%500==0 and fn:
            np.savez(model.modeldir+'/{}_{}_{}_{}_{}_{}_{}.npy'.format(fn, 0.1, 0.3, 0.3, 0.001, 5, -2),
                     log_err=out_dict['log_err'],
                     log_y_true=out_dict['log_y_true'],
                     log_y_pred=out_dict['log_y_pred'],
                     log_y_pred_ref=out_dict['log_y_pred_ref'],
                     log_idx=out_dict['log_idx'])


    return out_dict


def main(*args):

    if 0:
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        testing_x = mnist.test.images.reshape(10000, 28, 28, 1)
        testing_y = mnist.test.labels
    else:
        aa = np.load('/home/exx/Documents/Hope/generative_classifier/adversarial_attacks/FGSM_images/MNIST28/MNIST_FGSMeps0.4_and_binarized_examples.npz')
        testing_x = np.expand_dims(aa['FGSM_features'],3)
        testing_y = aa['orig_target']

    para_list = [0.2, 0.3, 0.6, 0.005, 5.0, -2.0]
    st = 0
    ed = len(testing_y)

    # Launch the graph
    FLAGS = tf.app.flags.FLAGS
    tfconfig = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
    )
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    with tf.variable_scope('VAE_subnet', reuse=False):
        model = VAE_subnet()
        model.build_training_model()
        # trained_model_ckpt = '/home/exx/Documents/Hope/generative_classifier/models/VAE/VAE_2018_02_12_22_10_27/experiment_111.ckpt'
        trained_model_ckpt = '/home/exx/Documents/Hope/generative_classifier/generative_classifier/models/VAE_sbunet/VAE_sbunet_2018_02_19_09_58_33/experiment_85.ckpt'
        saver = tf.train.Saver()
        saver.restore(sess, trained_model_ckpt)
        fn = "mnist_eps0.4" #"0.{}epsilon_{}to{}".format(args[0], st, ed)
        # build the classification optimization model
        apply_gradient_op, init_op = build_classify_model(model)
        out_dict = testing_classify_model(sess, model, apply_gradient_op, init_op, testing_x, testing_y, para_list, fn=fn)
        print('done')


if __name__ == "__main__":
    with tf.device('gpu:{}'.format(2)):
        main(3,0)