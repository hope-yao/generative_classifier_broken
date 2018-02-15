import tensorflow as tf
from utils.plot_figures import *



def get_attack_directions(model, sess):

    temp = set(tf.all_variables())

    input_pl = tf.placeholder(tf.float32, [1, model.imsize, model.imsize, 1], name='input_pl')
    input_pl_var = tf.Variable(input_pl, name='input_pl_var')
    label_pl = tf.placeholder(tf.float32, [1, model.num_class], name='label_pl')
    output = model.build_model(tf.tile(input_pl_var,(model.batch_size,1,1,1)), reuse=True)

    # find the gradient push away from true label
    negative_pred_err = -tf.reduce_mean(tf.abs(output['Predictions'] - label_pl))
    optimizer = tf.train.AdamOptimizer(0.01)
    grads = optimizer.compute_gradients(negative_pred_err, var_list=input_pl_var)
    apply_gradient_op = optimizer.apply_gradients(grads)
    gradient_direction = tf.gradients(negative_pred_err, input_pl_var)[0]
    init_op = tf.initialize_variables(set(tf.all_variables()) - temp)

    for i in range(10):
        batch_xs, batch_ys = model.mnist.test.next_batch(1)
        feed_dict = {input_pl: batch_xs.reshape(1, model.imsize, model.imsize, 1), label_pl:batch_ys}
        sess.run(init_op, feed_dict)

        delta_vec = {}
        delta_vec['x'] = np.squeeze(sess.run(gradient_direction, feed_dict))
        delta_vec['y'] = delta_vec['x'] * np.random.randn(*delta_vec['x'].shape)


        x_hist = []
        for i in range(50):
            _, adversarial_img = sess.run([apply_gradient_op, input_pl_var], feed_dict)
            x_hist += [adversarial_img]

        images = [np.squeeze(x_hist[-1]), batch_xs.reshape(model.imsize,model.imsize)]
        file_path = 'attacks.png'
        save_subplots(images, file_path)

        pred = {}
        pred['attacked'] = sess.run(output['Predictions'], {input_pl_var: x_hist[-1].reshape(1, model.imsize, model.imsize, 1)})[0]
        pred['original'] = sess.run(output['Predictions'], {input_pl_var: batch_xs.reshape(1, model.imsize, model.imsize, 1)})[0]


    return delta_vec, pred
