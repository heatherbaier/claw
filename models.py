import matplotlib.patches as patches
from torchvision import models, transforms
from collections import namedtuple, deque
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gym import Env, spaces
import PIL.Image as Image
import rasterio as rio
from torch import nn
import numpy as np 
import random
import torch
import time
import cv2 
import gym

from copy import deepcopy

device = "cpu"

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        self.mig_head = nn.Linear(linear_input_size, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))



class mig_model(torch.nn.Module):

    def __init__(self, resnet):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = torch.nn.Linear(in_features = 512, out_features = 1, bias = True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.flatten(start_dim=1)
        pred = self.fc(out)
        return out, pred




class lstm(torch.nn.Module):

    def __init__(self):
        super().__init__() 
        # self.rnn = torch.nn.LSTM(input_size = 512, hidden_size = 256, num_layers = 2, batch_first = True)
        
        self.fca = torch.nn.Linear(1024, 512)
        self.fcb = torch.nn.Linear(512, 256)
        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x, hidden = None):
        
        # print(torch.cat((x, torch.zeros(1, 1, 512)), dim = 2).shape)
        if hidden is None:
            out = self.fca(torch.cat((x, torch.zeros(1, 1, 512)), dim = 2))
            hidden = out.detach().clone()
        else:
            # print(hidden.shape)
            out = self.fca(torch.cat((x, hidden), dim = 2))
            hidden = out.detach().clone()
        # print(hidden)
        out = self.relu(out)
        out = self.fcb(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out, hidden




# rnn = torch.nn.LSTM(input_size = 512, hidden_size = 1, num_layers = 2, batch_first = True)
# input = torch.randn(1, 1, 512)
# h0 = torch.randn(2, 1, 1)
# c0 = torch.randn(2, 1, 1)
# output, (hn, cn) = rnn(input, (h0, c0))