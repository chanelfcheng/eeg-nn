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

    def __init__(self, h_size, n_layer, in_size, b_first = False, bidir = False):
        super(RNN, self).__init__()

        self.hidden_size = h_size
        self.num_layers = n_layer
        self.input_size = 5

        self.dict = {'Fr1': np.array([0, 3, 8, 7, 6, 5]), 'Fr2': np.array([ 2,  4, 10, 11, 12, 13]), 'Tp1': np.array([14, 23, 32, 41, 50]),
        'Tp2': np.array([22, 31, 40, 49, 56]), 'Cn1': np.array([15, 16, 17, 26, 25, 24, 33, 34, 35]), 'Cn2': np.array([21, 20, 19, 28, 29, 30, 39, 38, 37]),
        'Pr1': np.array([42, 43, 44, 52, 51]), 'Pr2': np.array([48, 47, 46, 54, 55]), 'Oc1': np.array([58, 57]), 'Oc2': np.array([60, 61])}

        self.batch_first = b_first
        self.bidirectional = bidir

        self.RNN_fL = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_fR = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_f = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_tL = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_tR = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_t = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_pL = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_pR = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_p = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_oL = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)
        self.RNN_oR = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.RNN_o = nn.RNN(self.hidden_size, self.hidden_size, self.num_layers, batch_first = self.batch_first, bidirectional = self.bidirectional)

        self.fc_f = nn.Sequential(
            nn.Linear(6*self.hidden_size, 16),
            nn.ReLU(),
            )

        self.fc_t = nn.Sequential(
            nn.Linear(5*self.hidden_size, 16),
            nn.ReLU(),
            )

        self.fc_p = nn.Sequential(
            nn.Linear(5*self.hidden_size, 16),
            nn.ReLU(),
            )

        self.fc_o = nn.Sequential(
            nn.Linear(2*self.hidden_size, 16),
            nn.ReLU(),
            )

        self.b_n1 = nn.BatchNorm2d(5)
        self.b_n2 = nn.BatchNorm1d(64)

    def forward(self, x):
        # Set initial states
        self.batch_size = x.shape[0]

        x = self.b_n1(x.permute(0,2,1).view(x.shape[0], 5, 1, -1 ))[:,:,0].permute(0,2,1)

        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

        k = list(self.dict.keys())

        fr_l = x[:, self.dict[k[0]]].permute(1, 0, 2)
        fr_r = x[:, self.dict[k[1]]].permute(1, 0, 2)

        tp_l = x[:, self.dict[k[2]]].permute(1, 0, 2)
        tp_r = x[:, self.dict[k[2]]].permute(1, 0, 2)

        p_l = x[:, self.dict[k[6]]].permute(1, 0, 2)
        p_r = x[:, self.dict[k[7]]].permute(1, 0, 2)

        o_l = x[:, self.dict[k[8]]].permute(1, 0, 2)
        o_r = x[:, self.dict[k[9]]].permute(1, 0, 2)

        x_fl, _ = self.RNN_fL(fr_l, h0)
        x_fr, _ = self.RNN_fR(fr_r, h0)

        x_tl, _ = self.RNN_tL(tp_l, h0)
        x_tr, _ = self.RNN_tR(tp_r, h0)

        x_pl, _ = self.RNN_tL(p_l, h0)
        x_pr, _ = self.RNN_tR(p_r, h0)

        x_ol, _ = self.RNN_oL(o_l, h0)
        x_or, _ = self.RNN_oR(o_r, h0)

        x_f = x_fr - x_fl
        x_t = x_tr - x_tl
        x_p = x_pr - x_pl
        x_o = x_or - x_ol

        x_f, _  = self.RNN_f(x_f, h0)
        x_t, _  = self.RNN_f(x_t, h0)
        x_p, _  = self.RNN_p(x_p, h0)
        x_o, _  = self.RNN_o(x_o, h0)

        x_f = x_f.permute(1, 0, 2)
        x_t = x_t.permute(1, 0, 2)
        x_p = x_p.permute(1, 0, 2)
        x_o = x_o.permute(1, 0, 2)

        x = torch.cat((self.fc_f(x_f.reshape(self.batch_size, -1)), self.fc_t(x_t.reshape(self.batch_size, -1)),
            self.fc_p(x_p.reshape(self.batch_size, -1)), self.fc_o(x_o.reshape(self.batch_size, -1))), dim=1)

        x = self.b_n2(x)
        #x = self.fc_f(x_f.reshape(self.batch_size, -1))  +  self.fc_t(x_t.reshape(self.batch_size, -1)) + self.fc_p(x_p.reshape(self.batch_size, -1)) + self.fc_o(x_o.reshape(self.batch_size, -1))
        #x = torch.cat((self.fc_f(x_f.reshape(self.batch_size, -1)), self.fc_t(x_t.reshape(self.batch_size, -1))), dim=1)
        x = x.reshape(self.batch_size, -1)
        #x = self.fc(x.reshape(self.batch_size, -1))
        return x