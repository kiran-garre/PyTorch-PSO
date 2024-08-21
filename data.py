import torch
from torch.utils.data import Dataset

def generate_points(f, num_dims, num_points, point_range):
    X = torch.FloatTensor(num_points, num_dims).uniform_(*point_range)
    Y = torch.tensor([f(*xi) for xi in X])
    return X, Y

class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)
