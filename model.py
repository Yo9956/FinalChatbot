import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, inputSize, hidSize, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(inputSize, hidSize) 
        self.l2 = nn.Linear(hidSize, hidSize) 
        self.l3 = nn.Linear(hidSize, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out