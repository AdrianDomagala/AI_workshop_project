import torch.nn as nn
import torch.nn.functional as F

class Titanic_Net(nn.Module):
    def __init__(self, input_size=6, num_classes=1):
        super(Titanic_Net, self).__init__()
        self.layer1 = nn.Linear(6, 32)
        self.layer2 = nn.Linear(32, 8)
        self.layer3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x





        