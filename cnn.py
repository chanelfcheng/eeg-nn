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

    def __init__(self, input_image=torch.zeros(1, 3, 32, 32), kernel=(3, 3), stride=1, padding=1, max_kernel=(2, 2),
                 n_classes=4):
        super(CNN, self).__init__()

        self.ClassifierCNN = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),

            Flatten(),

            nn.Linear(128,64)
        )

    def forward(self, x):
        x = self.ClassifierCNN(x)
        x = x.view(x.shape[0], -1)
        #x = self.ClassifierFC(x.view(x.shape[0], -1))
        return x

class Flatten(nn.Module):

    def forward(self, input):
        return input.view(input.size(0), -1)