import os
import numpy as np
from torch.utils.data import Dataset
import pickle

import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

class RML2016_Dataset(Dataset):
    def __init__(self, 
                 hdf5_file,  
                 snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18], 
                 lengths = 256, 
                 modulations = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'], 
                 one_hot = False,  
                 samples_per_key = None, 
                 ):
        self.file_path = hdf5_file
        self.modulations = modulations
        self.hot = one_hot
        self.sample_length = lengths
        self.data, self.length = [], []
        self.samples_per_key = int(samples_per_key) if samples_per_key else None
        self.modulation_map = dict(zip(modulations, range(len(modulations))))
        print("modulation map: {}".format(self.modulation_map))
        self.Xd = pickle.load(open(self.file_path, 'rb'), encoding='latin')
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], self.Xd.keys())))), [1, 0])
        self.X, self.lbl = [], []
        for mod in mods:
            for snr in snrs:
                for i in range(self.Xd[(mod, snr)].shape[0] if not self.samples_per_key else self.samples_per_key):
                    self.X.append(self.Xd[(mod, snr)][i])
                for i in range(self.Xd[(mod, snr)].shape[0] if not self.samples_per_key else self.samples_per_key):  
                    self.lbl.append((self.modulation_map[mod], snr))
        self.length = len(self.X)
        self.Xd = []

    def DataDealing(self, sig):
        re, im = np.real(sig), np.imag(sig)
        return np.array([re, im])
    
    def one_hot(self, label, total):
        return np.array([1 if i==label else 0 for i in range(total)])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        data = self.X[index]
        label, snr = self.lbl[index]
        return data, label, snr
        
