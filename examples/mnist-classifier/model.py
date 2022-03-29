import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F

class TinyModel(nn.Module):

    def __init__(self, 
                 input_channels: Optional[int] = 1, 
                 out_channels: Optional[int] = 10,
                 dropout_prob: Optional[float] = 0.2,
                 depth: Optional[int] = 2, 
                 width: Optional[int] = 128):
        super(TinyModel, self).__init__()
        layers = [nn.Conv2d(input_channels, width, 3, 1)]
        for layer in range(depth-1):
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(width, width, 3, 1))
            layers.append(nn.Dropout(dropout_prob))
        layers.append(nn.ReLU())
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(width, out_channels))
        self.sequence = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequence(x)
    
    def predict(self, x):
        with torch.no_grad():
            y = self.sequence(x)
            return F.softmax(y, dim=-1)
            