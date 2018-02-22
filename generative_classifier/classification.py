from VAE_subnet import *



def fast_build_classify_model(model):
    '''
    test a batch of images at once
    :param model:
    :return:
    '''
    temp = set(tf.all_variables())
    model.x_input_opt_in = tf.placeholder(tf.float32, [model.batch_size, model.imsize, model.imsize, 1], name='x_input_opt')
    model.y_input_opt_in = tf.placeholder(tf.float32, [model.batch_size, model.num_net], name='y_input_opt')
    model.y_input_opt = tf.Variable(model.y_input_opt_in)

    enc_z = model.encoder(model.x_input_opt_in, model.latent_dim * model.num_net, reuse=True)

    model.enc_z_sample, _ = mysample(enc_z, model.num_net, is_vae=True)  # just to split z value

    model.z_initial = tf.Variable(model.enc_z_sample)
    z_opt = []
    for i in range(model.num_net):
        z_opt += [model.z_initial[i]]
    model.out_sum_opt, model.out_sub_opt = model.decoder(z_opt, model.y_input_opt, reuse=True)
    model.gen_img = tf.sigmoid(model.out_sum_opt)

    model.bias = tf.placeholder(tf.float32, [], name='bias')
    model.theta = tf.placeholder(tf.float32, [], name='theta')
    model.c_wd = tf.placeholder(tf.float32, [], name='c_wd')

    model.gen_img_sig = tf.sigmoid(model.theta * model.gen_img + model.bias)
    model.x_ref = tf.sigmoid(model.theta * model.x_input_opt_in + model.bias)

    model.loss_opt = tf.reduce_mean(tf.abs(model.gen_img_sig - model.x_ref)) \
                     + model.c_wd * tf.norm(model.y_input_opt, ord=1)
    opt = tf.train.AdamOptimizer(0.01)

    grads_g = opt.compute_gradients(model.loss_opt, var_list=[model.y_input_opt, model.z_initial])
    apply_gradient_op = opt.apply_gradients(grads_g)

    init_op = tf.initialize_variables(set(tf.all_variables()) - temp)
    return apply_gradient_op, init_op


def joint_optimization(model, sess, para_list, x_input_opt, y_initial, apply_gradient_op, init_op):
    loss_t, y_l, y_u, c_wd, theta, bias = para_list

    feed_dict = {model.c_wd: c_wd,
                 model.theta: theta,
                 model.bias: bias,
                 model.x_input_opt_in: x_input_opt,
                 model.y_input_opt_in: y_initial}
    sess.run(init_op, feed_dict)

    y_pred_hist = []
    for i in range(50):
        _, y_pred = sess.run([apply_gradient_op, model.y_input_opt], feed_dict)
        y_pred_hist += [y_pred]

    # y_pred = np.clip(model.y_input_opt.eval(session=sess), 0, 1.49)
    z_pred = sess.run(model.z_initial, feed_dict)
    rec_err = np.mean(np.abs(sess.run(model.gen_img_sig - model.x_ref, feed_dict)),(1,2,3))

    return y_pred_hist, z_pred.transpose((1,0,2)), rec_err


def refine_classification(model, sess, x_input_opt, para_list, rec_err_batch, y_pred_batch, z_pred_batch):
    loss_t, y_l, y_u, c_wd, theta, bias = para_list

    mnist_idx = np.argsort(y_pred_batch,1)
    err_i = []
    yy_tt = []
    num = 3
    for i in range(num):
        yy = np.zeros_like(y_pred_batch)
        for j in range(32):
            yy[j, mnist_idx[j, -i - 1]] = 1
        feed_dict = {model.c_wd: c_wd,
                     model.theta: theta,
                     model.bias: bias,
                     model.x_input_opt_in: x_input_opt,
                     model.y_input_opt: yy}

        err_i += [np.mean(np.abs(sess.run(model.gen_img_sig - model.x_ref, feed_dict)), (1, 2, 3))]
        yy_tt += [yy]
    yy_tt = np.asarray(yy_tt)
    idx = np.argsort(np.asarray(err_i),0)
    y_refine = []
    for j in range(32):
        y_refine += [yy_tt[idx[0][j],j]]

    feed_dict = {model.c_wd: c_wd,
                 model.theta: theta,
                 model.bias: bias,
                 model.x_input_opt_in: x_input_opt,
                 model.y_input_opt: y_refine}
    rec_err_refine = np.mean(np.abs(sess.run(model.gen_img_sig - model.x_ref, feed_dict)), (1, 2, 3))
    z_pred_refine = z_pred_batch
    return np.asarray(y_refine), z_pred_refine, rec_err_refine


def fast_testing_classify_model(sess, model, apply_gradient_op, init_op, valid_x, valid_y, para_list, fn=None, img_idx=None):
    out_dict = {}

    # optimization
    out_dict['log_err'] = []

    n_epoch = len(valid_y) / model.batch_size
    for ep_i in range(n_epoch):
        opt_idx = np.arange(model.batch_size*ep_i, model.batch_size*(ep_i+1))
        y_true = valid_y[opt_idx]
        x_input_opt_noise = valid_x[opt_idx]
        # binarize first
        x_input_opt = np.round(x_input_opt_noise)
        # perform joint optimization
        y_initial =  model.batch_size * [[0.5] * 10]
        y_pred_hist, z_pred, rec_err = joint_optimization(model, sess, para_list, x_input_opt, y_initial, apply_gradient_op, init_op)
        y_pred = y_pred_hist[-1]
        # refine classification
        y_pred_refine, z_pred_refine, rec_err_refine = \
            refine_classification(model, sess, x_input_opt, para_list, rec_err, y_pred, z_pred)


        out_dict['log_idx'] = opt_idx if ep_i==0 else np.append(out_dict['log_idx'], opt_idx)
        out_dict['log_rec_err'] = rec_err if ep_i==0 else np.append(out_dict['log_rec_err'],rec_err)
        out_dict['log_rec_err_refine'] = rec_err_refine if ep_i==0 else np.append(out_dict['log_rec_err_refine'],rec_err_refine)
        out_dict['log_y_true'] = y_true if ep_i==0 else np.vstack([out_dict['log_y_true'],y_true])
        out_dict['log_y_pred'] = y_pred if ep_i==0 else np.vstack([out_dict['log_y_pred'],y_pred])
        out_dict['log_y_pred_refine'] = y_pred_refine if ep_i==0 else np.vstack([out_dict['log_y_pred_refine'],y_pred_refine])

        for i in range(model.batch_size):
            if (y_pred_refine[i] != y_true[i]).any() and fn:
                out_dict['log_err'] += [i + ep_i * model.batch_size]
                # save wrong cases
                img = sess.run(model.gen_img, {model.y_input_opt: y_pred, model.z_initial: z_pred.transpose(1, 0, 2)})
                plt.imshow(np.squeeze(img[i]), interpolation='None')
                plt.title('prediction: {}'.format(np.argmax(y_pred_refine)))  #
                plt.savefig(model.modeldir + '/idx{}_gen.png'.format(opt_idx))
                plt.imshow(np.squeeze(x_input_opt_noise[i]), interpolation='None')
                plt.title('rec_err: {}'.format(rec_err))  # prediction: {}
                plt.savefig(model.modeldir + '/idx{}_input.png'.format(opt_idx))

        if ep_i%500==0 and fn:
            np.savez(model.modeldir+'/{}_{}_{}_{}_{}_{}_{}.npy'.format(fn, 0.1, 0.3, 0.3, 0.001, 5, -2),
                     log_err=out_dict['log_err'],
                     log_y_true=out_dict['log_y_true'],
                     log_y_pred=out_dict['log_y_pred'],
                     log_y_pred_ref=out_dict['log_y_pred_refine'],
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
        trained_model_ckpt = '/home/exx/Documents/Hope/generative_classifier/assets/pretrained_generator/experiment_85.ckpt'
        saver = tf.train.Saver()
        saver.restore(sess, trained_model_ckpt)
        fn = "mnist_eps0.4" #"0.{}epsilon_{}to{}".format(args[0], st, ed)
        # build the classification optimization model
        apply_gradient_op, init_op = fast_build_classify_model(model)
        out_dict = fast_testing_classify_model(sess, model, apply_gradient_op, init_op, testing_x, testing_y, para_list, fn=fn)
        print('done')


if __name__ == "__main__":
    # Step0: configure the experiment
    cfg = {'resolution': 50,
           'scale': [5, .5],
           'lenet_ckpt': '/home/exx/Documents/Hope/generative_classifier/assets/pretrained_lenet/experiment_26.ckpt',
           'vaesub_ckpt': '/home/exx/Documents/Hope/generative_classifier/assets/pretrained_generator/experiment_85.ckpt',
           'gpu_idx': 3,
           }
    with tf.device('gpu:{}'.format(2)):
        main(3,0)