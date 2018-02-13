from VAE_subnet import *


def testing_model(sess, model, valid_x, valid_y, para_list, fn='mnist', gpu_idx=0, img_idx=None):
    loss_t, y_l, y_u, c_wd, theta, bias = para_list

    with tf.device('gpu:{}'.format(gpu_idx)):
        # build network
        temp = set(tf.all_variables())
        model.x_input_opt_in = tf.placeholder(tf.float32, [1, model.imsize, model.imsize, 1], name='x_input_opt')
        model.y_input_opt_in = tf.placeholder(tf.float32, [1, len(valid_y[0])], name='y_input_opt')
        model.x_input_opt = tf.tile(model.x_input_opt_in, (model.batch_size, 1, 1, 1))
        model.y_input_opt = tf.Variable(model.y_input_opt_in)

        enc_z = model.encoder(model.x_input_opt, model.latent_dim*model.num_net, reuse=True)

        model.enc_z_sample, _ = mysample(enc_z[0:1], model.num_net, is_vae=True) #just to split z value

        model.z_initial = tf.Variable(model.enc_z_sample)
        z_opt = []
        for i in range(model.num_net):
            z_opt += [tf.tile(model.z_initial[i], (model.batch_size, 1))]
        y_input_opt = tf.tile(model.y_input_opt, (model.batch_size, 1))
        model.out_sum_opt, model.out_sub_opt = model.decoder(z_opt, y_input_opt, reuse=True)

        model.bias = tf.placeholder(tf.float32, [], name='bias')
        model.theta = tf.placeholder(tf.float32, [], name='theta')
        model.c_wd = tf.placeholder(tf.float32, [], name='c_wd')

        model.out_sum_opt_sig = tf.sigmoid(model.theta * model.out_sum_opt + model.bias)
        model.x_ref = tf.sigmoid(model.theta * model.x_input_opt + model.bias)

        model.loss_opt = tf.reduce_mean(tf.abs(model.out_sum_opt_sig - model.x_ref)) \
                         + model.c_wd * tf.norm(model.y_input_opt)
        opt = tf.train.AdamOptimizer(0.01)
        # opt_sgd = tf.train.GradientDescentOptimizer(0.001)
        grads_g = opt.compute_gradients(model.loss_opt, var_list=[model.y_input_opt, model.z_initial])
        apply_gradient_opt = opt.apply_gradients(grads_g)

        def joint_optimization(x_input_opt):

            y0 = [0.5] * 10
            feed_dict = {model.c_wd: c_wd,
                         model.theta: theta,
                         model.bias: bias,
                         model.x_input_opt_in: np.expand_dims(x_input_opt, 0),
                         model.y_input_opt_in: np.expand_dims(y0, 0)}
            sess.run(tf.initialize_variables(set(tf.all_variables()) - temp), feed_dict)

            y_pred_hist = []
            for i in range(50):
                _, y_pred = sess.run([apply_gradient_opt,model.y_input_opt], feed_dict)
                y_pred_hist += [y_pred]

            # y_pred = np.clip(model.y_input_opt.eval(session=sess), 0, 1.49)
            z_pred = sess.run(model.z_initial, feed_dict)
            rec_err = sess.run(tf.reduce_mean(tf.abs(model.out_sum_opt_sig - model.x_ref)), feed_dict)

            return y_pred_hist, z_pred, rec_err

        def refine_classification(rec_err, y_pred, z_pred):
            loss_t = para_list[0]
            if rec_err > loss_t:
                y_refine = np.zeros_like(y_pred)
                z_pred_refine = np.zeros_like(z_pred)
                rec_err_refine = 0.
            elif not y_pred.any():
                y_refine = np.zeros_like(y_pred)
                z_pred_refine = np.zeros_like(z_pred)
                rec_err_refine = 0.
            else:
                # consider some special cases:
                mnist_idx = np.argsort(model.y_pred[0])
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

                    err_i += [np.mean(np.abs(sess.run(model.out_sum_opt_sig, feed_dict)[0, 0] - x_input_opt))]
                    yy_tt += [yy]
                idx = np.argsort(np.asarray(err_i))
                y_refine = yy_tt[idx[0]]

                rec_err_refine = sess.run(tf.reduce_mean(tf.abs(model.out_sum_opt_sig - model.x_ref)), feed_dict)

                # plt.imshow(self.sess.run(self.d_out_opt,
                #                          {self.y_input_opt: self.y_pred, self.z0: self.z_pred})[
                #                0, 0], interpolation='None')
                # plt.savefig('idx{}_gen.png'.format(opt_idx))
                z_pred_refine = z_pred
            return y_refine, z_pred_refine, rec_err_refine

        # optimization
        log_err = []
        log_idx = []
        log_y_pred = []
        log_y_pred_ref = []
        log_rec_err = []
        log_rec_err_ref = []
        log_y_true = []
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
            y_pred_hist, model.z_pred, model.rec_err = joint_optimization(x_input_opt)
            model.y_pred = y_pred_hist[-1]
            # refine classification
            model.y_pred_refine, model.z_pred_refine, model.rec_err_refine = \
                refine_classification(model.rec_err,model.y_pred, model.z_pred)

            log_idx += [opt_idx]
            log_y_true += [y_true]
            log_rec_err += [model.rec_err]
            log_y_pred += [model.y_pred]
            log_y_pred_ref += [model.y_pred_refine]
            log_rec_err_ref += [model.rec_err_refine]
            if int((np.asarray(model.y_pred_refine)-y_true).any()):
                log_err += [opt_idx]

            if opt_idx%500==0:
                np.savez(model.modeldir+'/{}_{}_{}_{}_{}_{}_{}.npy'.format(fn, 0.1, 0.3, 0.3, 0.001, 5, -2),
                         log_err=log_err, log_y_true=log_y_true, log_y_pred=log_y_pred, log_y_pred_ref=log_y_pred_ref,
                         log_idx=log_idx)
    return log_err


# plt.imshow(self.sess.run(self.d_out_opt,
#                          {self.y_input_opt: self.y_input_opt.eval(session=self.sess),
#                           self.z0: self.z_pred})[
#                0, 0], interpolation='None')
# plt.savefig('idx{}_gen.png'.format(opt_idx))
# return self.y_pred, self.y_pred, self.y_pred, self.y_pred, rec_err, rec_err

def main(*args):

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    testing_x = mnist.test.images.reshape(10000, 28, 28, 1)
    testing_y = mnist.test.labels
    para_list = [0.2, 0.3, 0.6, 0.001, 5.0, -2.0]
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

    model = generative_classifier()
    model.build_training_model()

    trained_model_ckpt = '/home/exx/Documents/Hope/generative_classifier/models/VAE/VAE_2018_02_12_22_10_27/experiment_111.ckpt'
    saver = tf.train.Saver()
    saver.restore(sess, trained_model_ckpt)
    fn = "mnist_testing" #"0.{}epsilon_{}to{}".format(args[0], st, ed)
    log_err = testing_model(sess, model, testing_x, testing_y, para_list, fn=fn, gpu_idx=args[0]-1)
    print('done')

# def testing_model(sess, model, valid_x, valid_y, para_list, name='mnist', gpu_idx=0, img_idx=None):

if __name__ == "__main__":
    main(1,0)