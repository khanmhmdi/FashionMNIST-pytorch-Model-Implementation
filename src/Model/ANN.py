import torch as nn
import torch.nn.functional as F
from torch import Module


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Module.Linear(784, 256)
        self.fc2 = Module.Linear(256, 128)
        self.fc3 = Module.Linear(128, 64)
        self.fc4 = Module.Linear(64, 10)
        # self.fc5 = nn.Linear(10, 10)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        output = self.fc4(x)

        return output
