# My Questions

## Sigmoid Output with MSE vs. Linear Output with MSE in Regression

The effect of the s-shape sigmoid scaled to 0-5 for the ratings maps the number line to 0-5, and makes the dot product result have the most differentiating power around value 0. For very large positive or negative values, they map to extreme ratings like 0 or 5, and varying them would not create much difference.

But one problem is that sigmoid output makes MSE wrt weigths non-convex! How to deal with that?

## Cross Entropy Loss and Maximum Likelihood

What is the deeper reason that regression (predicts real numbers) uses MSE loss and classification uses cross entropy loss? The classification optimization is a Maximum Likelihood Estimation but why?



