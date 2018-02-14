import numpy as np
import matplotlib.pyplot as plt

def plotting(classifier, figure_path):
    feed_dict = {
        classifier.X: classifier.mnist.test.images[:classifier.batch_size].reshape(classifier.batch_size, classifier.imsize, classifier.imsize, 1),
        classifier.label: classifier.mnist.test.labels[:classifier.batch_size]}
    # Applying encode and decode over test set
    encode_decode = classifier.sess.run(classifier.rec, feed_dict)
    f, a = plt.subplots(3, classifier.examples_to_show, figsize=(10, 2))
    for i in range(classifier.examples_to_show):
        a[0][i].imshow(np.reshape(classifier.mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        # a[2][i].imshow(np.reshape(mid_encode_decode[i], (28, 28)))
    f.show()
    f.savefig(figure_path)
    # plt.draw()
    # plt.show()