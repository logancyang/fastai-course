# Lesson 1 Pets

```py
# Shows the module imported from, function signature with types
# such as url:str, fname:Union[pathlib.Path, str]=None
# Union[...] means it can be any of the type in this list
help(func)

# python3 path object can be used like this
path_img = path/'image'

# ImageDataBunch in fastai has a factory method from_name_re() to use regex to get the data and labels
# Normalize to square image, size 224 is very commonly used
data = ImageDataBunch.from_name_re(
    path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
).normalize(imagenet_stats)
```

A `DataBunch` object in fastai contains everything about data. It has training, validation, test data, all the labels.

```py
# normalize the data in the DataBunch object to 0 mean 1 std
# in this case, RGB channels with be 0 mean and 1 std
data.normalize(imagenet_stats)

# Always take a look at the data
data.show_batch(rows=3, figsize=(7,6))

print(data.classes)
# data.c is the number of classes
len(data.classes), data.c
```

`Learner` in fastai is a model object, it takes in the `DataBunch` object and model architecture. It also let you specify the metric to look at for training.

```py
# Note that models.resnet34 is pretrained on imagenet
# error_rate is for validation set, because creating a DataBunch object automatically creates the validation set
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.model
# fit_one_cycle() is much better than fit() (paper in 2018), always use it. 4 is the # epoch to run, it's a good place to start
learn.fit_one_cycle(4)
# This can saves the model trained on different DataBunch to different places
learn.save('stage-1')
```

To interpret the output of a learner,

```py
# ClassificationInterpretation has a factory method from_learner,
# it takes in a learner object and returns a ClassificationInterpretation object
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()
len(data.valid_ds)==len(losses)==len(idxs)

# This is the most useful tool, it plots the examples the model got wrong
# with biggest losses, and show titles as
# `predicted / actual / score for predicted class / score for actual class`
interp.plot_top_losses(9, figsize=(15,11))

# Use doc() to check how to use, click `show in doc` to documentation website for more details and source code
doc(interp.plot_top_losses)

# Another useful tool, confusion matrix
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

# Yet another, the most useful, prints the most wrong ones (predicted, actual, # times)
interp.most_confused(min_val=2)
```

Previously we called `fit_one_cycle(4)`, **it only trains the last layer!!** The whole model is frozen by default!

*Note: transfer learning is always a two-step process, train the last layer for some epochs first, and then unfreeze the whole model for some more training/fine-tuning*

```py
# After training the last layer, we unfreeze the whole model
learn.unfreeze()
```

When we unfreeze the whole model and retrain, the error rate is actually worse than before.

Now, load the stage 1 model back on again, we need `lr_find` to find a good learning rate. The default learning rate is something like `0.003`. We need to change it.

The learning rate is the most important hyperparameter.

Use `learn.recorder.plot()` to plot `learning rate vs loss` graph and set the learning rate range to the steepest downward slope.

Next, unfreeze the model and use that learning rate

```py
learn.unfreeze()
# Pass a range of learning rate from 1e-6 to 1e-4
# means first layers use 1e-6 and last layers use 1e-4
# Good rule of thumb, set the right side of slice to be 10x smaller
# than default lr, and left to be at least 10x smaller than the lowest
# point in the lr vs loss graph
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
```

Now it gets better than stage 1!

With these two stages, the model can be already really good. It is at least average Kaggle practitioner level.

Next, we can use a bigger model - resnet50.

If GPU runs out of memory, use smaller batch size. It can be set in the DataBunch creation.

Again, the two-stage workflow

```py
data = ImageDataBunch.from_name_re(path_img, fnames, pat,
    ds_tfms=get_transforms(), size=299, bs=bs//2).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet50, metrics=error_rate)
# Stage 1
learn.fit_one_cycle(8)
learn.save('stage-1-50')
# Stage 2
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
# Pick learning rate range around the steepest downward slope
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))
# Evaluation
interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)
```

Note:
- For an imagenet-style dataset, it has data in folders named with the labels. MNIST for example, has images for `3` in a folder named `3`. Use `ImageDataBunch.from_folder(...)`
- For a csv file with columns `[filepath, label]`, use `ImageDataBunch.from_csv(...)`
- For data where labels are in the filename, use regex and `ImageDataBunch.from_name_re(...)`
- For more complex cases, construct any function and pass into `ImageDataBunch.from_name_func(...)`

What's really cool is that the documentation of fastai is actually a collection of Jupyter Notebooks with working example code!

Success story: a fastai alumnus created a security anomaly detection software by using the exact code for this lesson on mouse trajectories.

There are examples where problems can be converted into an image problem and apply CNN. Such as audio to spectral form.

## Homework 1

Use my own dataset to train an image classifier.

