import torch
import torch.nn as nn
import torch.optim as optim


softplus = torch.nn.Softplus()

class Meta_LR_Model(nn.Module):
    def __init__(self):
        super(Meta_LR_Model, self).__init__()

        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = softplus(self.fc1(x))
        x = softplus(self.fc2(x))
        x = softplus(self.fc3(x))
        x = softplus(self.fc4(x)) * 1e-3
        return x