# Lesson 7: Resnets from scratch; U-net; Generative adversarial networks; RNNs

Notebook: `lesson7-resnet-mnist`

PyTorch puts channel at the 1st dimension by default. An image in MNIST is a (1,28, 28) rank 3 tensor.

In fastai, there is a difference between validation and test set. Test set doesn't have label. Use validation set for model development. If you want to do inference on many things at a time and not one at a time, set the data as `test` instead of `valid`.

Initial steps:

1. Create ItemList from image folders
2. Split into `train` and `valid`.

*Tip: for data augmentation, MNIST can't have much transformation, you can't flip it because it changes the meaning of the number, you can't zoom because it's low res. The only transform is to add random padding. **Do this transform on training set but not validation set**.*

If not using a pretrained model, don't pass in `stat` in `normalize()` for databunch, it grabs a subset of the data at random and figures out how to normalize.

`plot_multi(_plot, nrow, ncol, figsize=())` is a fastai function that plots multiple examples in a grid. Define `_plot` first for what to show.

```py
# Showing a batch of data using the DataBlock API
xb,yb = data.one_batch()
xb.shape,yb.shape

# DataBunch has .show_batch()
data.show_batch(rows=3, figsize=(5,5))
```

Then we define the convolution function with fixed kernel size, stride and padding.

```py
# ni is # input channels, nf is # output filters (kernels)
def conv(ni,nf):
    return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)

model = nn.Sequential(
    # 1 channel in, 8 channels out (picked by us), output size 8*14*14
    conv(1, 8),
    nn.BatchNorm2d(8),
    nn.ReLU(),
    # 8 channel in, 16 channels out (picked by us), output size 16*7*7
    conv(8, 16), # 7
    nn.BatchNorm2d(16),
    nn.ReLU(),
    # 16 channel in, 32 channels out (picked by us), output size 32*4*4
    conv(16, 32), # 4
    nn.BatchNorm2d(32),
    nn.ReLU(),
    # 32 channel in, 16 channels out (picked by us), output size 16*2*2
    conv(32, 16), # 2
    nn.BatchNorm2d(16),
    nn.ReLU(),
    # 16 channel in, 10 channels out (picked by us), output size 10*1*1
    conv(16, 10), # 1
    nn.BatchNorm2d(10),
    Flatten()     # remove (1,1) grid
)
```

Trick: `Flatten()` gets rid of all the unit axes (the axes with 1s)! `(10, 1, 1)` becomes just `(10,)`, a flat vector of dim 10!

```py
# create learner
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
# print learner summary
print(learn.summary())
# pop data onto GPU
xb = xb.cuda()
# check model output shape
model(xb).shape
```

This is a model we built from scratch with a simple CNN architecture, it takes 12s to train on my GPU and got 98.8% accuracy! Already super good.

## Refactor

fastai has `conv_layer` so we can skip writing all the batch norm and relu's.

```py
def conv2(ni,nf): return conv_layer(ni,nf,stride=2)

model = nn.Sequential(
    conv2(1, 8),   # 14
    conv2(8, 16),  # 7
    conv2(16, 32), # 4
    conv2(32, 16), # 2
    conv2(16, 10), # 1
    Flatten()      # remove (1,1) grid
)

learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)

learn.fit_one_cycle(10, max_lr=0.1)
```

It's the same as previous code, just looks better. Train 10 epochs, we can get to 99%+ accuracy.

## Introduce the Residual Block

The residual block is a revolutionary technique in computer vision.

Kaiming He et. al. at Microsoft Research initially found that a 56-layer CNN was performing worse than a 20-layer CNN which made no sense. He created an architecture where a 56-layer CNN *contains* the 20-layer CNN, by adding some skip connections that skipped some conv layers. That way, it must be as least as good as the 20-layer CNN because the deeper CNN could just set the skipped conv layers to 0 and only keep the identity links.

<img src="./images/res-block.png" alt="Residual Block" align="middle"/>

Instead of having

```
output = conv( conv (x) )
```

The residual block is

```
output = conv( conv (x) ) + x
```

The result was that he won ImageNet that year (2015).

Trick: if an NN or GAN doesn't work so well, try replacing the conv layers with residual blocks!

Check out the fantastic paper [Visualizing the Loss Landscape](https://arxiv.org/abs/1712.09913). This is 3 years later since ResNet, and people started to realize why it worked. With the skip connections, the loss landscape is much smoother.

The batch norm had the same story. This reminds us *innovation usually comes from intuition*. Intuition comes first, people realize what's going on and why it works much later.

<img src="./images/loss-landscape.png" alt="Loss Landscape" align="middle"/>

fastai has `res_block`. We add a `res_block` after every `conv2` layer from previous code, we get

```py
model = nn.Sequential(
    conv2(1, 8),
    res_block(8),
    conv2(8, 16),
    res_block(16),
    conv2(16, 32),
    res_block(32),
    conv2(32, 16),
    res_block(16),
    conv2(16, 10),
    Flatten()
)
```

Further refactoring it,

```py
def conv_and_res(ni,nf):
    return nn.Sequential(conv2(ni, nf), res_block(nf))

model = nn.Sequential(
    conv_and_res(1, 8),
    conv_and_res(8, 16),
    conv_and_res(16, 32),
    conv_and_res(32, 16),
    conv2(16, 10),
    Flatten()
)

learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)

learn.lr_find(end_lr=100)
learn.recorder.plot()

learn.fit_one_cycle(12, max_lr=0.05)

print(learn.summary())
```

*Tip: when you try out new architectures, keep refactor the code and reuse more to **avoid mistakes**.*

Resnet is quite good and can reach SOTA accuracy for a lot of tasks. More modern techniques such as group convolutions don't train as fast.

*DenseNet* is another architecture, its only difference from Resnet is that instead of a `x + conv(conv(x))`, it does `concat(x, conv(conv(x)))` (the channel gets a little bigger). It is called a *DenseBlock* instead of ResBlock. The paper of DenseNet seems complicated but it's really very similar to Resnet.

DenseNet is very memory intensive because it maintains all previous features, BUT it has much fewer parameters. **It works really well for small datasets**.

## U-Net

Use resnet34 and half-stride. What half-stride is really doing is *nearest-neighbor interpolation* or a *bilinear interpolation* with stride 1, it up samples the patch and increases the size, as shown below.

<img src="./images/conv-upsample.png" alt="Conv UpSample" align="middle"/>

Fantastic paper for convolution: [A Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/pdf/1603.07285.pdf)

Nowadays we use a pretrained resnet34 as the *encoder* in U-net.

Trick: if you see two convs in a row, probably should use a resnet block instead. A skip connection with "+" or "concat" usually works great.

U-net came before resnet and densenet but it had a lot of the similar ideas and worked great for segmentation tasks.

*Tip: don't use U-net for classification, because you only need the down-sampling part, not the up-sampling part. Use U-net for generative purposes such as image segmentation because the output resolution is the same as input resolution.*

## Image Restoration with U-Net and GAN

Notebook: superres-gan

We use the U-net architecture to train a super-resolution model. This is a model which can increase the resolution of a low-quality image. Our model won't only increase resolution—it will also remove jpeg artifacts, and remove unwanted text watermarks.

In order to make our model produce high quality results, we need to create a custom loss function which incorporates *feature loss (also known as perceptual loss)*, along with *gram loss*. These techniques can be used for many other types of image generation task, such as image colorization.

Traditionally, the GAN is hard to train because the initial generator and critic are bad. Fastai uses pretrained generator and critic, so they are already pretty good. After that the training of GAN is much easier.

<img src="./images/fastai-gan.png" alt="GAN" align="middle"/>

To train a fastai version GAN, we need two folders, one with high-res original images, one with generated images.

Trick: free GPU memory without restarting notebook, run

```py
my_learner = None
gc.collect()
```

Running `nvidia-smi` won't show it freed because pytorch has pre-allocated cache, but it's available.

```py
# need to wrap the loss with AdaptiveLoss for GAN to work
# Will revisit in Part II
loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())
# Use gan_critic() and not resnet here
def create_critic_learner(data, metrics):
    return Learner(data, gan_critic(), metrics=metrics, loss_func=loss_critic, wd=wd)
# GAN version of accuracy: accuracy_thresh_expand
learn_critic = create_critic_learner(data_crit, accuracy_thresh_expand)

learn_critic.fit_one_cycle(6, 1e-3)
learn_critic.save('critic-pre2')
```

*Tip: for GAN, fastai's `GANLearner` figures out the back and forth training of the generator and the critic for us. Use the hyperparameters like this*

```py
switcher = partial(AdaptiveGANSwitcher, critic_thresh=0.65)
learn = GANLearner.from_learners(
    learn_gen, learn_crit,
    weights_gen=(1.,50.),
    show_img=False,
    switcher=switcher,
    opt_func=partial(optim.Adam, betas=(0.,0.99)),
    wd=wd
)

learn.callback_fns.append(partial(GANDiscriminativeLR, mult_lr=5.))

lr = 1e-4
learn.fit(40,lr)

# NOTE: the train_loss and gen_loss should stay around the same values
# because when the generator and critic both get better, the loss is relative.
# The only way to tell how it's doing is by looking at the image results
# Use show_img=True to check
```

### WGAN

Notebook `wgan` is briefly mentioned for the task of generating image from pure noise without pretraining. Jeremy mentioned it's a relatively old approach and the task isn't particularly useful, but it's good research exercise.

### Perceptual Loss (Feature Loss)

Notebook: `lesson7-superres`

Paper: [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf)

Jeremy didn't like the name "perceptual loss" and he named it "feature loss" in fastai library.

Convention: in U-net shaped architecture, the down-sampling part is called *encoder* and the up-sampling part is called *decoder*.

The paper's idea is to compare the generated image with the target image using a new loss function, that is, the activation from a middle layer in an ImageNet pretrained VGG network. Use the two images and pass them through this network up to that layer and check the difference. The intuition for this is that, each pixel in that activation should be capturing some feature of ImageNet images, such as furriness, round shaped, has eyeballs, etc. If the two images agree on these features they should have small loss with this loss function.

<img src="./images/feature-loss.png" alt="Feature Loss" align="middle"/>

With 1 GPU and 1-2hr time, we can generate medium res images from low res images, or high res from medium res using this approach.

A fastai student Jason in 2018 cohort created the famous [deOldify](https://github.com/jantic/DeOldify) project. He crappified color images to black and white, and trained *a GAN with feature loss* to color 19th century images!

## Recap

<img src="./images/recap.png" alt="recap" align="middle"/>

Watch the videos again and go through notebooks in detail to understand better.

## Recurrent Neural Network

Notebook: `lesson7-human-numbers`

Toy example dataset with numbers in English, the task is to predict the next word -- language model.

`xxbos`: beginning of stream, meaning start of document.

`data.bptt`: `bptt` is backprop thru time

Basic NN with 1 hidden layer:

<img src="./images/basic-nn.png" alt="Basic NN diagram" align="middle"/>

One step toward RNN

<img src="./images/basic-rnn.png" alt="To RNN" align="middle"/>

Basic RNN:

<img src="./images/basic-rnnn.png" alt="Basic RNN" align="middle"/>

Refactor, make it a loop --> RNN

<img src="./images/rnn.png" alt="RNN" align="middle"/>

There is nothing new for an RNN, it is just a fully connected NN with maintained states.

What GRU or LSTM is basically doing is to determine how much of the green arrow to keep and how much of the brown arrow to keep. Will see more in Course Part II.

<img src="./images/end.png" alt="The end" align="middle"/>

## Homework

- Read papers and write blog posts in plain language.
- Build visualizations and apps. Just finish something.

