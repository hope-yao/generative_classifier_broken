# Generative classifier
Making use of generative models to defend adversarial attacks. 
Here Variational Auto-Encoder with multiple decoders is used to test the idea.
Network structure:

![acc](../master/assets/net_struct.png)

## Prerequisite
Tensorflow 1.1.0

## Usage

### setting up
You can find pretrained models in 
/assets/pretrained_generator and /assets/pretrained_lenet

You can also train the discriminator from scratch, by running:

`python ./original_classifier/lenet_mnist.py` 

And to train the generator, by running:

`python ./generative_classifier/VAE_subnet.py` 

To monitor the training process, run:

```
$ tensorboard --logdir=./saved_logs
```

### perform classification

To obtain classification accuracy on MNIST/ adversarial images, run:

`python ./generative_classifier/classification.py` 

To obtain church window plot, run:

`python ./adversarial_attacks/plot_church_window.py` 


## Results
Here is a comparison of the overall accuracy with [binary filtered classifier](https://arxiv.org/abs/1704.01155):

![overall_acc](../master/assets/imgs/overall_acc.png)

[Church window plot](https://arxiv.org/abs/1611.02770) of this attack:

![church_plt_lenet](../master/assets/imgs/church_plt_lenet.png)

![church_plt_vaesub](../master/assets/imgs/church_plt_vaesub.png)
 
Critical FGSM attacked images:

![critical_img_lenet](../master/assets/imgs/critical_img_lenet.png)

![critical_img_vaesub](../master/assets/imgs/critical_img_vaesub.png)

## ToDO

-[] weight sharing in decoders

-[] smallNORB dataset

-[] overlapping mnist 