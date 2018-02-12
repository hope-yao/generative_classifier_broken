
def mnist36(mnist, bs, test=0):

    import numpy as np
    import random
    if test:
        all_images = mnist.test.images
        all_labels = mnist.test.labels
    else:
        all_images = mnist.train.images
        all_labels = mnist.train.labels
    data = []
    label = []
    imgsize = 36
    from tqdm import tqdm
    for j in range(bs):
        new_img_i = np.zeros((imgsize, imgsize))
        label_0 = 0
        label_1 = 0
        while label_0 == label_1:
            idx = np.random.choice(len(all_images), 2, replace=False)
            label_0 = np.argmax(all_labels[idx[0], :])
            label_1 = np.argmax(all_labels[idx[1], :])
        img_0 = all_images[idx[0]].reshape(28, 28)
        img_1 = all_images[idx[1]].reshape(28, 28)
        ul_coner_x_0 = random.randint(0, 8)
        ul_coner_y_0 = random.randint(0, 8)
        ul_coner_x_1 = random.randint(0, 8)
        ul_coner_y_1 = random.randint(0, 8)
        new_label = [label_0, label_1]
        new_img_i[ul_coner_x_0:ul_coner_x_0 + 28, ul_coner_y_0:ul_coner_y_0 + 28] += img_0
        new_img_i[ul_coner_x_1:ul_coner_x_1 + 28, ul_coner_y_1:ul_coner_y_1 + 28] += img_1
        label = label + [new_label]
        data = data + [new_img_i]

    data = 255.* np.expand_dims(np.asarray(data),1)  # magnitude from 0 to 2*255.
    label = np.asarray(label)
    label_onehot = np.zeros((bs, 10))
    label_onehot[np.arange(len(label)), label[:, 0]] = 1
    label_onehot[np.arange(len(label)), label[:, 1]] = 1

    return data,label_onehot
