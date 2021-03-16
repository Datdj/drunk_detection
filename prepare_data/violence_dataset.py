import sys
sys.path.append('prepare_data')
from torch.utils.data import Dataset
import numpy as np
import pickle
from video import Person
import random

class ViolenceDataset(Dataset):
    def __init__(self, fight_file, non_fight_file, series_length, min_num_poses):
        
        # Read binary file
        with open(fight_file, 'rb') as src:
            fight = pickle.load(src)
        with open(non_fight_file, 'rb') as src:
            non_fight = pickle.load(src)

        # Remove people that can not generate valid series
        for person in fight:
            person.get_valid_frames(series_length, min_num_poses)
        fight = [person for person in fight if len(person.valid_frames) > 0]
        for person in non_fight:
            person.get_valid_frames(series_length, min_num_poses)
        non_fight = [person for person in non_fight if len(person.valid_frames) > 0]
        
        # Store data
        self.series_length = series_length
        self.min_num_poses = min_num_poses
        num_fight = len(fight)
        self.data = fight + non_fight
        self.label = np.zeros(len(self.data))
        self.label[:num_fight] = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        person = self.data[index]
        start = random.choice(person.valid_frames)
        series = np.asarray([person.frames[person.mapping[i]].pose if i in person.mapping.keys() else np.zeros((13, 3)) for i in range(start, start + self.series_length)])
        series = series.reshape((self.series_length, -1))
        return series, self.label[index]

class ViolenceValDataset(ViolenceDataset):
    def __init__(self, fight_file, non_fight_file, series_length, min_num_poses):
        super().__init__(fight_file, non_fight_file, series_length, min_num_poses)
        self.series_idx = []
        for i in range(len(self.data)):
            person = self.data[i]
            for start in person.valid_frames:
                self.series_idx.append((i, start, self.label[i]))
    
    def __len__(self):
        return len(self.series_idx)

    def __getitem__(self, index):
        person = self.data[self.series_idx[index][0]]
        start = self.series_idx[index][1]
        series = np.asarray([person.frames[person.mapping[i]].pose if i in person.mapping.keys() else np.zeros((13, 3)) for i in range(start, start + self.series_length)])
        series = series.reshape((self.series_length, -1))
        label = self.series_idx[index][2]
        return series, label

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