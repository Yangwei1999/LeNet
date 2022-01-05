import torch
from torch import nn


class My_LeNet(nn.Module):
    def __init__(self):
        super(My_LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5),padding=2)
        self.sigmoid = nn.Sigmoid()
        self.pool1 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5))
        self.pool2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=(5,5))

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(120,84)
        self.linear2 = nn.Linear(84,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.sigmoid(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


