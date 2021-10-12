import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os

class ProcessedDataset(Dataset):
    def __init__(self, path_source, debug_subset=False, transform=None):
        super(ProcessedDataset, self).__init__()
        self.path_source = path_source
        data_temp = os.listdir(path_source)
        self.data = [i for i in data_temp if 'data_' in i]
        if debug_subset:
            self.data = [i for idx, i in enumerate(self.data) if idx % 5 == 0]


    def __getitem__(self, idx):
        x = torch.load(f'{self.path_source}/{self.data[idx]}')
        return x

    def __len__(self):
        return len(self.data)

