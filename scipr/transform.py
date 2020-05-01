import numpy as np
import torch
import torch.nn as nn

def fit_transform_rigid(A, B):
    # See http://nghiaho.com/?page_id=671
    # center
    A_centroid = np.mean(A, axis=0)
    B_centroid = np.mean(B, axis=0)

    H = np.dot((A - A_centroid).T, B - B_centroid)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    detR = np.linalg.det(R)
    x = np.identity(Vt.T.shape[1])
    x[x.shape[0]-1, x.shape[1]-1] = detR
    R = np.linalg.multi_dot([Vt.T, x, U.T])

    t = B_centroid.T - np.dot(R, A_centroid.T)

    return R, t
    
def fit_transform_affine(A, B, optim='adam', lr=1e-3, epochs=1000):
    d = A.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device {}'.format(device))
    A = torch.from_numpy(A).float().to(device)
    B = torch.from_numpy(B).float().to(device)
    f = torch.nn.Sequential()
    f.add_module('lin', torch.nn.Linear(d, d, bias=True))
    f.to(device)
    if optim == 'adam':
        optimizer = torch.optim.Adam(f.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(f.parameters(), lr=lr, momentum=0.9, nesterov=True)
    f.train()
    for e in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean(torch.norm(f(A) - B, p=2, dim=1)**2)
        if e % 100 == 0:
            print(f'\tEpoch: {e}/{epochs}, loss: {loss.item()}')
        loss.backward()
        optimizer.step()
    theta = np.zeros((d + 1, d + 1))
    theta[:d, :d] = f[0].weight.data.cpu().numpy()
    theta[:d, -1] = f[0].bias.data.cpu().numpy()
    theta[-1, -1] = 1.
    return theta, f[0].weight.data.cpu().numpy(), f[0].bias.data.cpu().numpy()

activations = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': nn.LeakyReLU
}

class Autoencoder(nn.Module):
    def __init__(self, input_size, layer_sizes, act='tanh', dropout=0.0, batch_norm=False, last_layer_linear=False, tie_weights=True):
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
                self.encoder.add_module('enc_dropout_{}'.format(layer), nn.Dropout(p=self.dropout))
            # Linearity
            self.encoder.add_module('enc_lin_{}'.format(layer), nn.Linear(prev_size, size))
            # BN
            if self.batch_norm:
                self.encoder.add_module('enc_batch_norm_{}'.format(layer), nn.BatchNorm1d(size))
            # Finally, non-linearity
            self.encoder.add_module('enc_{}_{}'.format(act, layer), activations[act]())
            prev_size = size
                
        reversed_layer_list = list(self.encoder.named_modules())[::-1]
        decode_layer_count = 0
        for name, module in reversed_layer_list:
            if 'lin_' in name:
                size = module.weight.data.size()[1]
                if self.dropout > 0:
                    self.decoder.add_module('dec_dropout_{}'.format(decode_layer_count), nn.Dropout(p=self.dropout))
                # Linearity
                linearity = nn.Linear(prev_size, size)
                if self.tie_weights:
                    linearity.weight.data = module.weight.data.transpose(0, 1)
                self.decoder.add_module('dec_lin_{}'.format(decode_layer_count), linearity)
                # if decode_layer_count < len(self.layer_sizes) - 1:
                # if True:
                if not (decode_layer_count == (len(self.layer_sizes) - 1) and self.last_layer_linear):
                    # BN
                    if self.batch_norm:
                        self.decoder.add_module('dec_batch_norm_{}'.format(decode_layer_count), nn.BatchNorm1d(size))
                    # Finally, non-linearity
                    self.decoder.add_module('dec_{}_{}'.format(act, decode_layer_count), activations[act]())
                prev_size = size
                decode_layer_count += 1
                        
    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

def fit_transform_autoencoder(A, B, hidden_sizes=[64], act='leaky_relu', optim='adam', lr=1e-3, epochs=1000):
    d = A.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device {}'.format(device))
    A = torch.from_numpy(A).float().to(device)
    B = torch.from_numpy(B).float().to(device)

    f = Autoencoder(input_size=d, layer_sizes=hidden_sizes, act=act, dropout=0.0, batch_norm=False, last_layer_linear=False, tie_weights=True)
    f.to(device)
    if optim == 'adam':
        optimizer = torch.optim.Adam(f.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(f.parameters(), lr=lr, momentum=0.9, nesterov=True)
    f.train()
    for e in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean(torch.norm(f(A) - B, p=2, dim=1)**2)
        if e % 100 == 0:
            print(f'\tEpoch: {e}/{epochs}, loss: {loss.item()}')
        loss.backward()
        optimizer.step()
    return f
