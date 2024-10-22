import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ВАШ КОД ЗДЕСЬ
        # определите слои сети
        # размер исходной картинки 32х32

        # слой 1
        self.conv1 = nn.Conv2d(in_channels=3, 
                               out_channels=3, #колво фильтров
                               kernel_size=(5,5)) #28x28
        # pool
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2)) #14x14 
        #слой 2
        self.conv2 = nn.Conv2d(in_channels=3, 
                               out_channels=5, 
                               kernel_size=(3,3)) #12
        # poll
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2)) #6x6

        self.flatten = nn.Flatten()
        
        # linear 1
        self.fc1 = nn.Linear(6*6*5, 100 )
        self.fc2 = nn.Linear(100,10 )


    def forward(self, x):
        # ВАШ КОД ЗДЕСЬ
        # реализуйте forward pass сети
        # размерность х ~ [64, 3, 32, 32]
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.flatten(x) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

def create_model():
    return ConvNet()
