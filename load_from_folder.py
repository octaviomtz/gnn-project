import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os

class ProcessedDataset(Dataset):
    def __init__(self, path_source, transform=None):
        super(ProcessedDataset, self).__init__()
        self.path_source = path_source
        self.data = os.listdir(path_source)

    def __getitem__(self, idx):
        x = torch.load(f'{self.path_source}/{self.data[idx]}')
        return x

    def __len__(self):
        return len(self.data)

