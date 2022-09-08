import torch
import glob
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torch.optim as optim

from torch.autograd import Function

device = torch.device('cuda:0')

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, device):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.device = device

        self.conv1 = nn.Conv1d(1, 32, 3, 1)
        self.conv2 = nn.Conv1d(32, 64, 3, 1)
        self.conv3 = nn.Conv1d(64, 128, 3, 1)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = x.view(-1, 1, self.input_dim)
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool1d(x, 2)
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x