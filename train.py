import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import rasterio as rio
import gym
import random

from gym import Env, spaces
import time

import matplotlib.patches as patches

from models import *

from torchvision import models, transforms

from earth_env import *

from IPython import display

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

torch.autograd.set_detect_anomaly(True)



env = EarthObs(num_channels = 3, num_actions = 5)



for epoch in range(0, 5):


    obs = env.reset()

    # last_screen = env.get_screen()
    current_screen = env.get_screen()

    memory = ReplayMemory(10000)

    # while True:

    done = False

    while not done:

        # Select and perform an action
        action = env.select_action()

        # Save current state so we can push it to memory in a couple lines
        current_state = env.view_box.clip_image(cv2.imread("./test_image.png"))

        _, reward, done, _ = env.step(action.item())

        next_state = env.view_box.clip_image(cv2.imread("./test_image.png"))

        memory.push(current_state, action, next_state, reward)

        # print(action)

        env.render()

        # Perform one step of the optimization (on the policy network)
        env.optimize_model()

        print("\n")


    # env.close()