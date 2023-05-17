import torch
from torch import nn


class MLP(nn.Module):
    """
    Multi-layer perceptron
    """

    def __init__(self, input_dim, hidden_dims, out_dim):
        """
        hidden_dims is a list which contain the number of neurons in each layer
        """
        super().__init__()

        modules = []
        for in_size, out_size in zip([input_dim] + hidden_dims, hidden_dims):
            modules.append(nn.Linear(in_size, out_size))
            modules.append(nn.LayerNorm(out_size))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_dims[-1], out_dim))
        self.fc = nn.Sequential(*modules)

    def forward(self, inputs):
        """
        forward pass
        """
        fcs = self.fc(inputs)
        #print('--->Shape of FCs<---: ', fcs.shape)
        return fcs.reshape(-1,1)
