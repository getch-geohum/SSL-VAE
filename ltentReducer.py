import torch
from torch import nn
import numpy as np
import os

class ReducerNet(nn.Module):
    def __init__(self, feature_dim=100, save=False):
        self.feature_dim = feature_dim
        self.save = save
        self.in_dim = 256
        self.conv = nn.Conv2d(self.in_dim, 100)
        self.pool = nn.MaxPooling2D()
        self.dnn = nn.dnn()
        self.relu = nn.Relu()

    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dnn(x.reshape(x.shape[0],-))
        x = self.relu(x)

        return x
