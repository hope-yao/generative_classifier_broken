import tensorflow as tf
from utils.plot_figures import *



def get_attack_directions(model, sess):

    temp = set(tf.all_variables())

    input_pl = tf.placeholder(tf.float32, [1, model.imsize, model.imsize, 1], name='input_pl')
    input_pl_var = tf.Variable(input_pl, name='input_pl_var')
    label_pl = tf.placeholder(tf.float32, [1, model.num_class], name='label_pl')
    output = model.build_model(tf.tile(input_pl_var,(model.batch_size,1,1,1)), reuse=True)

    # find the gradient push away from true label
    negative_pred_err = tf.reduce_mean(tf.abs(output['Predictions'] * label_pl))
    optimizer = tf.train.AdamOptimizer(0.01)
    grads = optimizer.compute_gradients(negative_pred_err, var_list=input_pl_var)
    apply_gradient_op = optimizer.apply_gradients(grads)
    gradient_direction = tf.gradients(negative_pred_err, input_pl_var)[0]
    init_op = tf.initialize_variables(set(tf.all_variables()) - temp)

    orth_transform = np.asarray([[(-1) ** j] * model.imsize for j in range(model.imsize)])

    for i in range(10):
        batch_xs, batch_ys = model.mnist.test.next_batch(1)
        batch_xs = batch_xs.reshape(model.imsize,model.imsize)
        feed_dict = {input_pl: batch_xs.reshape(1, model.imsize, model.imsize, 1), label_pl:batch_ys}
        sess.run(init_op, feed_dict)

        delta_vec = {}
        gradient_direction_var = np.squeeze(sess.run(gradient_direction, feed_dict))
        c_grad = np.sum(batch_xs) / np.sum(np.abs(gradient_direction_var))
        delta_vec['x'] = c_grad * gradient_direction_var
        delta_vec['y'] = delta_vec['x'][::-1, ::-1] * orth_transform

        x_hist = []
        for i in range(50):
            _, adversarial_img = sess.run([apply_gradient_op, input_pl_var], feed_dict)
            x_hist += [adversarial_img]

        images = [np.squeeze(x_hist[-1]), batch_xs]
        file_path = 'attacks.png'
        save_subplots(images, file_path)

        pred = {}
        pred['attacked'] = sess.run(output['Predictions'], {input_pl_var: x_hist[-1].reshape(1, model.imsize, model.imsize, 1)})[0]
        pred['original'] = sess.run(output['Predictions'], {input_pl_var: batch_xs.reshape(1, model.imsize, model.imsize, 1)})[0]
        feed_dict_step = {input_pl_var: (delta_vec['x'] + batch_xs).reshape(1, model.imsize, model.imsize, 1)}
        pred['one_large_step'] = sess.run(output['Predictions'], feed_dict_step)[0]

        # if attack is successful, plot church plot
        resolution = 1000
        test_res = np.zeros((resolution,resolution))
        axis_range = [-20.,20., -20.,20.]
        if (np.round(pred['attacked']) != np.round(pred['original'])).any():
            for i in range(0,resolution,1):
                for j in range(0,resolution,1):
                    w_x = axis_range[0] + (axis_range[1] - axis_range[0]) * i/resolution
                    w_y = axis_range[2] + (axis_range[3] - axis_range[2]) * j/resolution
                    test_img = batch_xs + w_x*delta_vec['x'] + w_y*delta_vec['y']
                    test_pred = sess.run(output['Predictions'], {input_pl_var: test_img.reshape(1, 28, 28, 1)})[0]
                    test_res[i, j] = np.argmax(test_pred)
            church_plt(test_res.transpose(), axis_range) #transpose becasue matrix and image different axis notation

    return delta_vec, pred
