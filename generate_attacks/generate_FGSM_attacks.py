import tensorflow as tf


def get_attack_directions(model, sess):

    opt = tf.train.AdamOptimizer(0.01)
    grads_g = opt.compute_gradients(model.loss_opt, var_list=[model.y_input_opt, model.z_initial])
    apply_gradient_op = opt.apply_gradients(grads_g)

    # init_op = tf.initialize_variables(set(tf.all_variables()) - temp)
    init_op = tf.initialize_variables(model.input_opt)

    return
