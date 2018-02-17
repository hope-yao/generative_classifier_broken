# Generative classifier
![acc](../master/assets/net_struct.png)

## Prerequisite
Tensorflow 1.1.0

## Usage

To train the generator, run:

`python ./generative_classifier/VAE_subnet.py` 


To monitor the training process, run:

```
$ tensorboard --logdir=./saved_logs
```

To perform classification, run:

`python ./generative_classifier/classification.py` 


## Results
Example of FGSM attacked images:

![attacks](../master/assets/attacks.png)

[Church window plot](https://arxiv.org/abs/1611.02770) of this attack:
![church_plt](../master/assets/church_plt.png)

Here is a comparison of the overall accuracy with [binary filtered classifier](https://arxiv.org/abs/1704.01155):
![overall_acc](../master/assets/overall_acc.png)



## ToDO
