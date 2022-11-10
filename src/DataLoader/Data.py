import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler  # for validation test


class Data:
    def __init__(self, train_path, test_path):
        """This class will load and make data
        readable for pytorch model."""

        self.train_path = train_path
        self.test_path = test_path

        self.train_data = None
        self.test_data = None

    def load_data(self):
        self.train_data = pd.read_csv(self.train_path)
        self.test_data = pd.read_csv(self.test_path)
        return self.train_path, self.test_path

    def visualize_random_image(self):
        return plt.imshow(np.array(self.train_data.loc[10][1:]).reshape(28, 28))

    def prepare_data(self, batch_size):
        """This function read csv data as tensor and
        return dataloader object for model input"""

        indices = list(range(len(self.train_data.index)))
        np.random.shuffle(indices)
        # to get 20% of the train set
        split = int(np.floor(0.2 * len(self.train_data)))
        train_sample = SubsetRandomSampler(indices[:split])
        valid_sample = SubsetRandomSampler(indices[split:])

        df = torch.from_numpy(np.array(self.train_data))
        # splitting label and data
        df = TensorDataset(df[:, 1:], df[:, 0])

        train = torch.utils.data.DataLoader(df, sampler=train_sample, batch_size=batch_size)
        valid = torch.utils.data.DataLoader(df, sampler=valid_sample, batch_size=batch_size)
        test = torch.utils.data.DataLoader(self.test_data)

        return train, valid, test
