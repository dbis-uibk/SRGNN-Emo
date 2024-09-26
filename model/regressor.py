import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_


class Regressor(nn.Module):

    def __init__(self, input_dim, dim, num_targets, dropout=0.0):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_targets),
        )

        self.reset_parameters()

    def forward(self, x):
        return self.layers(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()