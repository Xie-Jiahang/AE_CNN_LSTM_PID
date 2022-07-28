import os
import numpy as np
import scipy.io
import torch

from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import utils
from utils import GLO


class pair_DataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x_one = self.x[index]
        y_one = self.y[index]

        return (x_one, y_one)


class process_data():
    def __init__(self, _in, _target):
        self._in = _in
        self._target = _target

    def encap(self, num):
        batch_size = GLO.get_value('batch_size')
        train_index = GLO.get_value('train_index')
        test_index = GLO.get_value('test_index')

        # _in_square,_target_square=utils.save_adjusted(self._in,self._target,num)

        # transform = transforms.Compose([transforms.ToTensor()]) # will transpose the shape
        # training data
        train_input = self._in[list(train_index)]
        train_target = self._target[list(train_index)]
        train_input = torch.tensor(train_input)
        train_target = torch.tensor(train_target)
        train_dataset = pair_DataSet(train_input, train_target)
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)  # in pairs
        # test data
        test_input = self._in[list(test_index)]
        test_target = self._target[list(test_index)]
        test_input = torch.tensor(test_input)
        test_target = torch.tensor(test_target)
        test_dataset = pair_DataSet(test_input, test_target)
        test_dataloader = DataLoader(test_dataset, batch_size=len(
            test_index), shuffle=False, drop_last=False)  # in pairs

        return train_dataloader, test_dataloader
