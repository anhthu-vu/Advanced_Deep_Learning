In this pratical session, I implemented normalizing flow models. To be more specific:

- In `utils.py` file, I implemented:
  - a naive normalizing flow model with only one layer, where the decoder is an affine transformation defined as $x = f(z; s, t) = z*e^s + t$. The model is trained on a dataset containing samples drawn from the distribution N(-1., 1.5). The prior distribution is chosen to be N(0., 1.)
