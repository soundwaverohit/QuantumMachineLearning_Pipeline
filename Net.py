import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import qiskit
from Hybrid import Hybrid 


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self._conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self._conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self._dropout = nn.Dropout2d()
        self._fc1 = nn.Linear(256, 64)
        self._fc2 = nn.Linear(64, 1)
        self._hybrid = Hybrid(1,qiskit.Aer.get_backend('aer_simulator'), 100, np.pi/2)
        # The circuit measurement serves as the final prediction as provided by measurment
        
        
    def forward(self, x):
        x = F.relu(self._conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self._conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = self._dropout(x)
        
        x = x.view(1, -1)
        
        x = F.relu(self._fc1(x))
        x = self._fc2(x)
        
        x = self._hybrid(x)
        
        return torch.cat((x, 1-x), -1)
        