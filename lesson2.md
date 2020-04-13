# Lesson 2 Data Cleaning and Production. SGD from Scratch

The notebook "Lesson 2 Download" has code for downloading images from Google images search result in parallel. It is a great way to create my own image dataset.

Note: `ImageDataBunch` can create validation set with `valid_pct`. ALWAYS SET RANDOM SEED BEFORE CREATING THE VALIDATION SET. We need the same validation set for each run.

Other times, randomness is good because we need to know the model runs stably with randomness.

DL models are pretty good at randomly noisy data. But "biased noisy" data is not good.

```py
# Fastai way of opening an image for inspection
img = open_image(path/'black'/'00000021.jpg')
```

## Facilitate experiments with notebook GUI widgets

There is an in-notebook GUI widget called `fastai.widgets.ImageCleaner` for cleaning the dataset. It has buttons to remove problematic examples and creates a new cleaned.csv from where you can create a new ImageDataBunch with the corrected labels to continue training your model.

These things are called `ipywidgets`.

## Put models in production

Inference on CPU is good enough for the vast majority of use cases. Inference on GPU is a big hassle, you have to deal with batching and queueing, etc. Unless the website has extremely high traffic, inferece on CPU is a much better choice, and it can be horizontally scaled easily.

```py
defaults.device = torch.device('cpu')
# Open image for inspection
img = open_image(path/'black'/'00000021.jpg')
img
```

The code below loads the trained model and is run at web app starting time once, should run fairly quickly.

```py
# We create our Learner in production enviromnent like this, just make sure that path contains the file 'export.pkl' from before
learn = load_learner(path)
```

Then we can do inference in the web application,

```
# inference
pred_class, pred_idx, outputs = learn.predict(img)
pred_class
```

The endpoint should look like (this example is for Starlette app)

```py
@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    img = open_image(BytesIO(bytes))
    _,_,losses = learner.predict(img)
    return JSONResponse({
        "predictions": sorted(
            zip(cat_learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })
```

Note: Starlette is similar to Flask but supports modern `async` and `await` for Python 3. FastAPI is another option.

## Common Training Problems

- Learning rate
- Number of epochs

If validation loss explodes, learning rate is too high. **Decrease the learning rate.**

If training loss is higher than validation loss, it means **underfitting**, either learning rate is too low or number of epochs too low. **Try higher learning rate or more epochs**.

If the **error rate goes down and up again**, it's probably **overfitting**. But it's pretty hard to overfit with the settings in fastai here. In this case, if the learning rate is already good and the model is trained a long time but we are still not satisfied with the error rate (note fastai error rate is always validation error), it can mean that we **need more data**. There is no shortcut to know how much data we need in advance. But sometimes we don't need that much data we thought we needed.

*Note: Some say if training loss is lower than validation loss, it's overfitting. IT IS NOT TRUE!! Any model that's trained correctly always have training loss lower than validation loss. That is right.*

*Another note: unbalanced classes is often NOT a problem, it just works. Jeremy spent years experimenting and thinks it's not a problem, no need to do oversampling for less common classes, it just works.*

Rule of thumb for training flow and setting learning rate

```py
default_lr = 3e-3
learn.fit_one_cycle(4, default_lr)
learn.unfreeze()
learn.fit_one_cycle(4, slice(steepest_pt_in_plot, default_lr/10))
```

In Lesson 2 SGD notebook, we generate some data points for a linear regression.

```py
n=100
# tensor n by 2, all ones
x = torch.ones(n,2)
# In PyTorch, any function that ends with _ means no return and modify in-place!
# With ., int becomes float
x[:,0].uniform_(-1.,1)
x[:5]
a = tensor(3.,2)
# Note in Python '@' is matmult. It's more general tensor product in pytorch
y = x@a + torch.rand(n)
```

The above is 95% we need to know about PyTorch
- create a tensor
- update a tensor
- tensor product

Note: The `rank` of a tensor is the number of dimensions, or axes. An RGB image is a rank 3 tensor. A vector is a rank 1 tensor. Tensor is just high dimensional arrays.

Trick: for `matplotlib` animations, set `rc('animation', html='jshtml')` instead of `%matplotlib notebook`.

Vocab

- learning rate: step size to multiply gradient by
- epoch: one complete run on all data. A batch size of 100 for 1000 datapoints means 1 epoch = 10 iterations. In genral we don't want to do too many epochs, it lets the model see the same data too many times and overfit
- minibatch: random bunch of data points to do weight update
- SGD: gradient descent using minibatches
- Model architecture: the math function to fit the data with
- Parameters: or weights or coefficients. The knobs to tune for a model
- Loss function: how far away to the target

## Homework

- Make logistic regression optimization animation and blog post
- Make a minimal web app for a model using FastAPI
