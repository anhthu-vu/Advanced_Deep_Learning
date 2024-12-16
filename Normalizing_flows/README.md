In this pratical session, I implemented normalizing flow models. To be more specific:

- In `utils.py` file, I implemented:
  - a naive model with only one layer, where the decoder is an affine transformation defined as $x = f(z; s, t) = z*e^s + t$. The model is trained on a dataset containing samples drawn from the distribution $\mathcal{N}(-1., 1.5)$. The prior distribution is chosen to be $\mathcal{N}(0., 1.)$.
  - the Glow model and trained it on Scikit-learn's [moon](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) dataset. The prior distribution is chosen to be a multivariate Gaussian $\mathcal{N}(0, I_2)$, where I_2 is the 2x2 identity matrix.

- 
