from torch.utils.data import Dataset
import numpy as np
import pickle
from video import Person

class ViolenceDataset(Dataset):
    def __init__(self, fight_file, non_fight_file):
        with open(fight_file, 'rb') as src:
            fight = pickle.load(src)
        with open(non_fight_file, 'rb') as src:
            non_fight = pickle.load(src)
        num_fight = len(fight)
        self.data = fight.extend(non_fight)
        self.label = np.zeros(len(self.data))
        self.label[:num_fight] = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

class ViolenceDataset_old_version(Dataset):
    '''
        Args:
            - fight_file: the path to fight.npy (see generate_data.py)
            - non_fight_file: the path to non_fight.npy (see generate_data.py)
    '''

    def __init__(self, fight_file, non_fight_file):
        fight = np.load(fight_file)
        num_fight = fight.shape[0]
        non_fight = np.load(non_fight_file)
        self.data = np.concatenate((fight, non_fight), axis=0)
        shape = self.data.shape
        self.data = self.data.reshape(shape[0], shape[1], shape[2] * shape[3])
        self.label = np.zeros(self.data.shape[0])
        self.label[:num_fight] = 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.label[index]