import torch
import torch.nn as nn

from .base import Transformer

activations = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU
}


class Autoencoder(nn.Module):
    def __init__(self, input_size, layer_sizes, act='tanh', dropout=0.0,
                 batch_norm=False, last_layer_linear=False, tie_weights=True):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.act = activations[act]
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.last_layer_linear = last_layer_linear
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
                   self.last_layer_linear):
                    # BN
                    if self.batch_norm:
                        self.decoder.add_module(
                            'dec_batch_norm_{}'.format(decode_layer_count),
                            nn.BatchNorm1d(size))
                    # Finally, non-linearity
                    self.decoder.add_module(
                        'dec_{}_{}'.format(act, decode_layer_count),
                        activations[act]())
                prev_size = size
                decode_layer_count += 1

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


class StackedAutoEncoder(Transformer):
    def __init__(self, hidden_sizes=[64], act='leaky_relu', optim='adam',
                 lr=1e-3, epochs=1000):
        self.hidden_sizes = hidden_sizes
        self.act = act
        self.optim = optim
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device {}'.format(self.device))
        super().__init__()

    def fit(self, A, B):
        d = A.shape[1]
        A = torch.from_numpy(A).float().to(self.device)
        B = torch.from_numpy(B).float().to(self.device)
        f = Autoencoder(input_size=d, layer_sizes=self.hidden_sizes,
                        act=self.act, dropout=0.0, batch_norm=False,
                        last_layer_linear=False, tie_weights=True)
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
                print(f'\tEpoch: {e}/{self.epochs}, loss: {loss.item()}')
            loss.backward()
            optimizer.step()
        model = {
            'autoencoder': f
        }
        return model

    def transform(self, model, A):
        A = torch.from_numpy(A).float().to(self.device)
        # self.autoencoders_.eval()
        model['autoencoder'].eval()
        A = model['autoencoder'].forward(A)
        A = A.detach().cpu().numpy()
        return A

    def chain(self, step_model, step_number):
        if self.model is None:
            self.model = {'autoencoder': nn.Sequential()}
        self.model['autoencoder'].add_module(f'autoencoder_{step_number}',
                                             step_model['autoencoder'])
        print(self.model)

    def finalize(self, A_orig, A_final):
        pass
