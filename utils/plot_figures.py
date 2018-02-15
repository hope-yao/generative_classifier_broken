import seaborn as sns
from pylab import *
import matplotlib.gridspec as gridspec
import numpy as np


def visualize_generator_performance(classifier, figure_path):
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

def confidence_plot():

    aa=np.load('/home/hope-yao/Downloads/figdata.npz')
    idx = 3#3 6 12 16
    confs = aa['classifications'][idx]
    imgs = aa['features'][idx]


    plt.figure(figsize=(13,8), dpi=80)
    G = gridspec.GridSpec(5, 9)
    ###
    axes_cmp = subplot(G[3:, :])
    x = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    #change#
    y1 = confs[0]
    y2 = confs[1]
    y3 = []
    for i in range(9):
        aa=np.load('/home/hope-yao/Downloads/cmp_mnist/{}.npz'.format(0.05*i))['pred']
        y3 += [aa[9]/np.sum(aa)]


    plt.plot(x, y1, color='k', linestyle='dashed', marker='v',
         markerfacecolor='k', markersize=10, label="no defense")
    plt.plot(x, y2, color='r', linestyle='dashed', marker='o',
         markerfacecolor='r', markersize=10, label="binarize defense")
    plt.plot(x, y3, color='b', linestyle='dashed', marker='*',
         markerfacecolor='b', markersize=10, label="generative defense")
    plt.legend(loc='lower left', frameon=False, fontsize=18)
    plt.xlim(-0.02, 0.42)
    plt.ylim(-0.05, 1.1)
    #x = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    #plt.xticks(x, ('0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4'), fontsize = 20)
    x = [0, 0.1, 0.2, 0.3, 0.4]
    plt.xticks(x, ('0', '0.1', '0.2', '0.3', '0.4'), fontsize = 20)
    plt.xlabel("FGSM noise level", fontsize=25)
    plt.ylabel("Confidence", fontsize=25)

    ###
    axes_orig = []
    orig = imgs[0]
    for i in range(9):
        axes_orig += [subplot(G[0,i])]
        xticks([]), yticks([])
        plt.imshow(orig[i],interpolation='none')
        if i==0:
            plt.ylabel("Adversarial", fontsize=18)

    ###
    axes_bf = []
    bf = imgs[1]
    for i in range(9):
        axes_bf += [subplot(G[1,i])]
        xticks([]), yticks([])
        plt.imshow(bf[i],interpolation='none')
        if i==0:
            plt.ylabel("binarized", fontsize=18)

    ###
    axes_gen = []
    for i in range(9):
        aa=np.load('/home/hope-yao/Downloads/cmp_mnist/{}.npz'.format(0.05*i))
        gen = aa['img']
        axes_gen += [subplot(G[2,i])]
        xticks([]), yticks([])
        plt.imshow(gen,interpolation='none')
        if i==0:
            plt.ylabel("generation", fontsize=18)

    #plt.savefig('../figures/confidence.png', dpi=64)
    show()

def save_subplots(images, file_path):
    num = len(images)
    plt.figure(figsize=(4,8), dpi=80)
    G = gridspec.GridSpec(1, num)
    axes_gen = []

    for i, img in enumerate(images):
        axes_gen += [subplot(G[0,i])]
        xticks([]), yticks([])
        plt.imshow(img,interpolation='none')
    plt.show()
    plt.savefig(file_path)


