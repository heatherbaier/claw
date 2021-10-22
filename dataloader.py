from torchvision import models, transforms
from collections import namedtuple, deque
from ViewBox import *
import torch
import json
import cv2
import os

from earth_env import *


class Dataset():

    def __init__(self, batch_size, imagery_dir, json_path):

        self.batch_size = batch_size
        self.imagery_dir = imagery_dir

        with open(json_path, "r") as f:
            self.ys = json.load(f)

        self.Batch = namedtuple('Batch', ('images', 'environment', 'ys'))

        self.data = []
        self.to_tens = transforms.ToTensor()
        self.load_data()


    def load_data(self):

        count = 0

        for impath in os.listdir(self.imagery_dir):

            # If have counted up to the amount of a batch, reset all of the lists
            if (count % self.batch_size) == 0:
                # ims, envs, ys = [], [], []
                cur_data = []

            # Load in and prep all of the data
            im = cv2.imread(os.path.join(self.imagery_dir, impath))

            cur_data.append((os.path.join(self.imagery_dir, impath), self.ys[impath.replace(".png", "")], EarthObs(impath = os.path.join(self.imagery_dir, impath), y_val = self.ys[impath.replace(".png", "")], num_channels = 3, num_actions = 5, display = False)))

            # If we are at the last element before we reset the lists, create a Batch 
            # namedtuple from our current lists and append them to our dataset
            if (count % self.batch_size) == self.batch_size - 1:
                self.data.append(cur_data)

            count += 1




"""
A TENSOR HAS SHAPE B, C, H, W
"""