import numpy as np
import pickle

def load_data(npz_file):
    data = np.load(npz_file)
    X = pickle.loads(data['data'])
    y = pickle.loads(data['label'])
    return X, y