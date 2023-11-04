import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from emnist_prediction.constants import IMG_SIZE

class HandwritingsDataset(Dataset):

    def __init__(self, X_file_path, y_file_path,
                 sample_transform=None, data_transform=None):
        # TODO: add some checks for file existance? number of samples?
        self.X_file_path = X_file_path
        self.y_file_path = y_file_path
        self.sample_transform = sample_transform

        self.X_data = np.load(X_file_path)
        y_data = np.load(y_file_path)

        if data_transform:
            self.X_data = data_transform(self.X_data)

        self.y_labels = y_data.argmax(axis=-1)
        self.X_data = torch.from_numpy(self.X_data).to(torch.float32)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, index):
        # Here we could load this lazily
        if self.sample_transform:
            return self.sample_transform([self.X_data[index], self.y_labels[index]])
        return self.X_data[index], self.y_labels[index]


class HandwritingsBalancedDataset(HandwritingsDataset):

    def __init__(self, sampling_algorithm, **kwargs):
        super().__init__(**kwargs)
        self.sampling_algorithm = sampling_algorithm
        self.transform_X_y()

    def transform_X_y(self):

        # Data for the resampler should be in format (n_samples, n_features)
        X = self.X_data.reshape(self.X_data.shape[0], -1)
        X, self.y_labels = self.sampling_algorithm.fit_resample(X, self.y_labels)
        self.X_data = X.reshape(-1, *IMG_SIZE)
        # Back to torch tensors 
        self.X_data = torch.from_numpy(self.X_data).to(torch.float32)