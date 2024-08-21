import torch
import torch.nn as nn
import math

# weights_init_uniform_() was taken from https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch

class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )
        self.model.apply(self.weights_init_uniform_)

    def weights_init_uniform_(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0 / math.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

    def forward(self, x):
        return self.model(x)
