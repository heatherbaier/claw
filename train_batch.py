from torchvision import models, transforms
import torch
import json
import cv2
import os

from dataloader import *
from ViewBox import *
from trainer import *
from utils import *

ds = Dataset(1, "./test_ims/", "./migration_data.json")

print(len(ds.data))

# print(ds.data)



n_actions = 5
device = "cpu"

policy_net = DQN(128, 128, n_actions).to(device)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(policy_net.parameters(), lr = 0.01)
memory = ReplayMemory(10000)


trainer = Trainer(num_channels = 3,
                  num_actions = 5,
                  model = policy_net,
                  criterion = criterion,
                  optimizer = optimizer,
                  memory = memory,
                  Transition = Transition,
                  train_dl = ds.data,
                  device = "cpu",
                  max_grabs = 5)

trainer.train()

# print(len(ds.data))