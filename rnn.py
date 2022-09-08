import torch
import glob
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torch.optim as optim

from torch.autograd import Function

device = torch.device('cuda:0')

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.device = device

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = x.view(-1, 1, self.input_dim)
        x, _ = self.gru(x)
        x = x.contiguous().view(-1, self.hidden_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x