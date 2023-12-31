import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from cam import loadstack


# Custom dataset
class FlowDataset(Dataset):
    def __init__(self,
                 data_index_list,
                 data,
                 targets_index_list=None,
                 targets=None,
                 transform=None,
                 train=True):
        """
        The dataset read the images on the fly
        Args:
            data_index_list: shuffled index list for training image path
            data (string list): A list of training image path 
            targets_index_list (string list): shuffled index list for test image path
            targets: A list of test image path
            train: True train mode, false eval mode (print data path)
        """
        self.data_index_list = data_index_list
        self.data = data
        self.targets_index_list = targets_index_list
        self.targets = targets
        self.count = 0
        self.train = train

    def eval(self):
        self.train = False

    def __len__(self):
        return len(self.data_index_list)

    def __getitem__(self, idx):
        #TODO implement a new dataset using the stack 
        sample_ind = self.data_index_list[idx]
        if self.targets_index_list is not None:
            label_ind = self.targets_index_list[idx]

        sample_1 = np.asarray(Image.open(
            self.data[0][sample_ind])) * 1.0 / 255.0
        sample_2 = np.asarray(Image.open(
            self.data[1][sample_ind])) * 1.0 / 255.0

        if not self.train:
            print(' ')
            print('img1: ', self.data[0][sample_ind])
            print('img2: ', self.data[1][sample_ind])
            if self.targets is not None:
                print('label: ', self.targets[sample_ind])
            print(' ')

        sample = torch.FloatTensor([sample_1, sample_2])
        # read_flow convert the RGB value in .flo to velocity components u and v
        """
        if self.targets is not None:
            label = fz.read_flow(self.targets[label_ind])
            label = torch.FloatTensor([label[..., 0], label[..., 1]])
            return sample, label
        """
        return sample
    
class FlowDataset(Dataset):
    def __init__(self,
                 filename,index):
        """
        Dataset for flow image
        Parameters
        ----------
        filename : str
            file path for the stack
        index : array
            numpy array with indices to analyse
        """
        self.stack = loadstack(filename)
        self.index = index
        
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        idx = self.index[idx]
        sample_1 = self.stack[idx].data
        sample_2 = self.stack[idx + 1].data

        sample = torch.FloatTensor([sample_1, sample_2])
        
        return sample