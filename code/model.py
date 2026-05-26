"""RAE model definition.

By default the encoder/decoder are single linear layers (no biases on the
hidden path, no activations), matching the architecture described in the
paper. Hidden layers can be added via ``hidden_dims`` if desired, but the
theoretical analysis in the paper assumes the linear case.
"""
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=None,
                 activation='tanh', dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        hidden_dims = hidden_dims or []

        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'sigmoid':
            act = nn.Sigmoid()
        else:
            act = nn.Tanh()

        # Encoder
        enc_layers = []
        prev = input_dim
        if hidden_dims:
            for h in hidden_dims:
                enc_layers += [nn.Linear(prev, h), nn.LayerNorm(h), act, nn.Dropout(dropout)]
                prev = h
            enc_layers.append(nn.Linear(prev, output_dim))
        else:
            enc_layers.append(nn.Linear(input_dim, output_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirrored)
        dec_layers = []
        prev = output_dim
        if hidden_dims:
            for h in reversed(hidden_dims):
                dec_layers += [nn.Linear(prev, h), nn.LayerNorm(h), act, nn.Dropout(dropout)]
                prev = h
            dec_layers.append(nn.Linear(prev, input_dim))
        else:
            dec_layers.append(nn.Linear(output_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z)
