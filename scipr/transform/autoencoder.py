import logging

import torch
import torch.nn as nn

from .base import Transformer

activations = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU
}


class _Autoencoder(nn.Module):
    def __init__(self, input_size, layer_sizes, act='tanh', dropout=0.0,
                 batch_norm=False, last_layer_act='tanh', tie_weights=True):
        super(_Autoencoder, self).__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.act = act
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.last_layer_act = last_layer_act
        self.tie_weights = tie_weights

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        prev_size = self.input_size
        for layer, size in enumerate(layer_sizes):
            # Apply dropout
            if self.dropout > 0:
                self.encoder.add_module('enc_dropout_{}'.format(layer),
                                        nn.Dropout(p=self.dropout))
            # Linearity
            self.encoder.add_module('enc_lin_{}'.format(layer),
                                    nn.Linear(prev_size, size))
            # BN
            if self.batch_norm:
                self.encoder.add_module('enc_batch_norm_{}'.format(layer),
                                        nn.BatchNorm1d(size))
            # Finally, non-linearity
            self.encoder.add_module('enc_{}_{}'.format(act, layer),
                                    activations[act]())
            prev_size = size

        reversed_layer_list = list(self.encoder.named_modules())[::-1]
        decode_layer_count = 0
        for name, module in reversed_layer_list:
            if 'lin_' in name:
                size = module.weight.data.size()[1]
                if self.dropout > 0:
                    self.decoder.add_module(
                        'dec_dropout_{}'.format(decode_layer_count),
                        nn.Dropout(p=self.dropout))
                # Linearity
                linearity = nn.Linear(prev_size, size)
                if self.tie_weights:
                    linearity.weight.data = module.weight.data.transpose(0, 1)
                self.decoder.add_module(
                    'dec_lin_{}'.format(decode_layer_count), linearity)
                # if decode_layer_count < len(self.layer_sizes) - 1:
                # if True:
                if not (decode_layer_count == (len(self.layer_sizes) - 1) and
                   self.last_layer_act is None):
                    # BN
                    if self.batch_norm:
                        self.decoder.add_module(
                            'dec_batch_norm_{}'.format(decode_layer_count),
                            nn.BatchNorm1d(size))
                    # Finally, non-linearity
                    self.decoder.add_module(
                        'dec_{}_{}'.format(last_layer_act,
                                           decode_layer_count),
                        activations[last_layer_act]())
                prev_size = size
                decode_layer_count += 1

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


class StackedAutoEncoder(Transformer):
    """Use multiple autoencoders to align the pairs of cells.

    Fit a "stack" of autoencoders, the output of one feeding in as the input
    into the next one (thereby "composing" them). At each step of SCIPR, the
    next autoencoder is fitted, and then added onto the overall stack. Since
    these are autoencoders, the output dimensions of each are the same as the
    input, so the dimensions are maintained.

    Will automatically search for and use a GPU if available.

    Parameters
    ----------
    hidden_sizes : list of int
        The sizes (widths) of the hidden layers of each autoencoder. This is
        one side of the "funnel" of the autoencoder architecture, and the other
        side is built to be symmetric (same as ``hidden_sizes`` but in
        reverse).

    act : {'leaky_relu', 'relu', 'sigmoid', 'tanh'}
        Which non-linear activation function to use for the autoencoders.

    last_layer_act : {None, 'leaky_relu', 'relu', 'sigmoid', 'tanh'}
        Which non-linear activation function to use for the final layer
        (output) of the autoencoders. ``None`` means no non-linearity. See
        Warnings below.

    optim : {'adam', 'sgd'}
        Which torch optimizer to use.

    lr : float
        Learning rate to use in gradient descent.

    epochs : int
        Number of iterations to run gradient descent.

    Warnings
    --------
    Be aware that your choice of input normalization to SCIPR might have
    implications for your choice of the ``last_layer_act`` parameter. For
    exmaple, If your input normalization scales your input features to
    [0, 1.0], then you may want your final layer to use a sigmoid activation.
    Or if your input normalization allows for input features to be (-\\infty,
    +\\infty), then having `None` as the last layer's activation might be best.
    This is just something to keep in mind, ultimately you may choose what
    performs the best alignment for you.
    """
    def __init__(self, hidden_sizes=[64], act='leaky_relu',
                 last_layer_act=None, optim='adam', lr=1e-3, epochs=1000):
        self.hidden_sizes = hidden_sizes
        self.act = act
        self.last_layer_act = last_layer_act
        self.optim = optim
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        log = logging.getLogger(__name__)
        log.info(f'Using device {self.device}')
        super().__init__()

    def fit(self, A, B):
        log = logging.getLogger(__name__)
        d = A.shape[1]
        A = torch.from_numpy(A).float().to(self.device)
        B = torch.from_numpy(B).float().to(self.device)
        f = _Autoencoder(input_size=d, layer_sizes=self.hidden_sizes,
                         act=self.act, dropout=0.0, batch_norm=False,
                         last_layer_act=self.last_layer_act, tie_weights=True)
        f.to(self.device)
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
        model = {
            'autoencoder': f
        }
        return model

    def transform(self, model, A):
        A = torch.from_numpy(A).float().to(self.device)
        model['autoencoder'].eval()
        A = model['autoencoder'].forward(A)
        A = A.detach().cpu().numpy()
        return A

    def chain(self, model, step_model, step_number):
        if model is None:
            model = {'autoencoder': nn.Sequential()}
        model['autoencoder'].add_module(f'autoencoder_{step_number}',
                                        step_model['autoencoder'])
        return model

    def finalize(self, model, A_orig, A_final):
        log = logging.getLogger(__name__)
        log.info('Final model:\n' + str(model))
        return model
