import torch.nn as nn
import torch.nn.functional as F


class ANN(nn.Module):
    def __init__(self , dropout=False):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, 10)
        self.dropout = nn.Dropout(0.35)
        # self.fc5 = nn.Linear(10, 10)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = (F.relu(self.fc2(x)))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = (F.relu(self.fc4(x)))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        if self.dropout:
            x = self.dropout(x)
        x = F.relu(self.fc7(x))

        output = self.fc8(x)

        return output

