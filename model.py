import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch

from cnn import CNN
from rnn import RNN
from load_npz import load_data
from utils import create_if_not_exists

# SEED-V dataset
eeg_input_dim = 310
eye_input_dim = 33
output_dim = 12

def main():
    print("Loading data...")
    X, y = load_data("./data/eeg_data_sep/1_123.npz")
    print("X: ", type(X))
    print("y: ", type(y))

    cnn = CNN(eeg_input_dim, [32, 64, 128], output_dim, 5, "cuda:0")

if __name__ == "__main__":
    main()