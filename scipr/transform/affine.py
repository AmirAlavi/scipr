import logging

import numpy as np
import torch

from .base import Transformer


class Affine(Transformer):
    """ Use an affine transformation function to align the pairs of cells.

    The affine function is of the form ``f(x) = Wx + b``, where ``W`` and ``b``
    are the learnable weights. ``W`` has the shape (genes, genes) and ``b``
    (the bias term) has shape (genes,).

    Parameters
    ----------
    optim : {'adam', 'sgd'}
        Which torch optimizer to use.

    lr : float
        Learning rate to use in gradient descent.

    epochs : int
        Number of iterations to run gradient descent.
    """
    def __init__(self, optim='adam', lr=1e-3, epochs=1000):
        self.optim = optim
        self.lr = lr
        self.epochs = epochs
        super().__init__()

    def fit(self, A, B):
        log = logging.getLogger(__name__)
        d = A.shape[1]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f'Using device {device}')
        A = torch.from_numpy(A).float().to(device)
        B = torch.from_numpy(B).float().to(device)
        f = torch.nn.Sequential()
        f.add_module('lin', torch.nn.Linear(d, d, bias=True))
        f.to(device)
        if self.optim == 'adam':
            optimizer = torch.optim.Adam(f.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.SGD(f.parameters(), lr=self.lr,
                                        momentum=0.9, nesterov=True)
        f.train()
        for e in range(self.epochs):
            optimizer.zero_grad()
            loss = torch.mean(torch.norm(f(A) - B, p=2, dim=1)**2)
            if e % 100 == 0:
                log.info(f'\tEpoch: {e}/{self.epochs}, loss: {loss.item()}')
            loss.backward()
            optimizer.step()
        # theta is the augmented matrix wich includes weights W and bias in
        # a single matrix
        theta = np.zeros((d + 1, d + 1))
        theta[:d, :d] = f[0].weight.data.cpu().numpy()
        theta[:d, -1] = f[0].bias.data.cpu().numpy()
        theta[-1, -1] = 1.

        model = {
            'theta': theta,
        }
        return model

    def transform(self, model, A):
        d = A.shape[1]
        W = model['theta'][:d, :d]
        bias = model['theta'][:d, -1]
        return np.dot(W, A.T).T + bias

    def chain(self, model, step_model, step_number):
        # Affine functions can be composed easily when represented in their
        # augmented matrix form, simply left multiply their (augmented)
        # transformation matrices by each other.
        if model is None:
            return step_model
        else:
            model['theta'] = np.dot(step_model['theta'],
                                    model['theta'])
            return model

    def finalize(self, model, A_orig, A_final):
        # Since we've been updating the overal model in the chain function,
        # we don't need to do anything here.
        return model
